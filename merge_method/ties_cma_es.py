"""ties_cma_es.py — 用 CMA-ES 搜索 TIES 的逐任务 λ。

和 ties_merging.py 的区别：
  1. 模型只加载一次，合并/回滚都在显存里做，搜索期间不落盘
  2. 目标函数 = 若干任务的平均准确率（--eval_max_samples 可做代理评测）
  3. 用 CMA-ES 在 λ 空间里搜索，边界 [--lo, --hi]
  4. 搜索结束用全量复核 top-k，只保存最好的那个

λ 的顺序和 --adapters 一致，例如: mnli,qnli,rte,qqp,mrpc,sst2,cola
对应 lams=[λ_mnli, λ_qnli, ...]。

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from task_arithmetic import open_source
from ties_merging import ties_merge
from common import evaluate, get_task, load_split, set_seed

TASK_ORDER = ["mnli", "qnli", "rte", "qqp", "mrpc", "sst2", "cola"]


def parse_args():
    p = argparse.ArgumentParser(description="用 CMA-ES 搜 TIES 的逐任务 λ")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapters", required=True, help="逗号分隔，顺序对应 λ")
    p.add_argument("--tasks", default=",".join(TASK_ORDER), help="逗号分隔，和 adapters 对齐")
    p.add_argument("--trim", type=float, default=0.2)
    p.add_argument("--skip_embed", action="store_true")
    p.add_argument("--max_modules", type=int, default=0, help="调试用，0=全部")

    # CMA-ES
    p.add_argument("--pop", type=int, default=5, help="每代采样数")
    p.add_argument("--gens", type=int, default=6, help="最多几代")
    p.add_argument("--sigma", type=float, default=0.3, help="初始步长")
    p.add_argument("--lo", type=float, default=0.3, help="λ 下界")
    p.add_argument("--hi", type=float, default=2.0, help="λ 上界")

    p.add_argument("--eval_max_samples", type=int, default=300, help="0=全量")
    p.add_argument("--holdout", default="", help="不参与目标函数、只做最后检验的任务")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--log", default=None)
    return p.parse_args()


class Merger:
    """模型常驻显存，apply/restore 都在内存里完成。"""

    def __init__(self, base, paths, trim, skip_embed, max_modules=0):
        self.sources = [open_source(p) for p in paths]

        # 判断模块集合是否一致，与ties_merging.py里的一样
        base_set = self.sources[0].names
        for s in self.sources[1:]:
            if s.names != base_set:
                raise ValueError(f"{s.path} 的模块集合与第一个不一致")

        self.trim = trim
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map={"": 0}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.named = dict(self.model.named_modules())

        modules = sorted(base_set & set(self.named))

        # 调试的时候用
        if skip_embed:
            modules = [m for m in modules if m not in ("model.embed_tokens", "lm_head")]
        if max_modules:
            modules = modules[:max_modules]
        self.modules = modules

        # 快照原始权重用于精确回滚：w - tau 会残留 bf16 舍入误差，几十次后会累积
        self.baseline = {
            n: self.named[n].weight.detach().to("cpu", torch.bfloat16).clone()
            for n in modules
        }

    def apply(self, lams):
        with torch.no_grad():
            # 用TIES跟新权值
            for name in self.modules:
                w = self.named[name].weight
                # TIES 是正齐次的，均匀权重放在前/后等价；非均匀必须在前面，否则没有意义
                deltas = [float(lam) * s.delta(name, w)
                          for lam, s in zip(lams, self.sources)]
                tau = ties_merge(deltas, keep=self.trim)
                w.add_(tau.to(w.device, w.dtype))

    def restore(self):
        with torch.no_grad():
            for name in self.modules:
                # 利用__init__保存的baseline恢复权值
                self.named[name].weight.copy_(self.baseline[name])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = [s.strip() for s in args.adapters.split(",") if s.strip()]
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    if len(paths) != len(tasks):
        raise ValueError(f"--adapters {len(paths)} 个，--tasks {len(tasks)} 个，必须一样多")

    holdout = args.holdout.strip()
    if holdout and holdout not in tasks:
        raise ValueError(f"--holdout={holdout} 不在 --tasks 里")

    print(f"[1/4] 加载 {len(paths)} 个 delta 源 + 底座 …", flush=True)

    # 创建一个Merger实例，模型常驻显存，apply/restore 都在内存里完成
    merger = Merger(args.base, paths, args.trim, args.skip_embed, args.max_modules)

    print("[2/4] 加载评测数据 …", flush=True)

    datasets = []
    for name in tasks:
        # 获取任务的设置
        spec = get_task(name)
        datasets.append((spec, load_split(spec, spec.val_split, None)))
    
    # 评估函数
    def run_eval(lams, max_samples):
        merger.apply(lams)
        try:
            accs = {}
            for spec, ds in datasets:
                m = evaluate(merger.model, merger.tokenizer, spec, ds,
                             batch_size=args.batch_size, max_len=args.max_len,
                             max_samples=max_samples, show=0)
                accs[spec.name] = m["accuracy"]
        finally:
            merger.restore()          # 评测抛异常也要回滚，否则后续全错
        kept = [v for k, v in accs.items() if k != holdout]
        return sum(kept) / len(kept), accs

    history = []

    # 记录函数
    def record(lams, score, accs, stage):
        history.append({
            "stage": stage,
            "lams": [round(float(x), 4) for x in lams],
            "score": round(float(score), 5),
            "accs": {k: round(float(v), 5) for k, v in accs.items()},
        })
        extra = " ".join(f"{k}={v:.4f}" for k, v in accs.items())
        print(f"    score={score:.4f}  {extra}", flush=True)

    print("[3/4] 基线（λ 全 1）…", flush=True)
    t0 = time.time()
    s0, a0 = run_eval([1.0] * len(paths), args.eval_max_samples)
    record([1.0] * len(paths), s0, a0, "baseline")

    print(f"    用时 {time.time() - t0:.1f}s", flush=True)

    print(f"[4/4] CMA-ES（pop={args.pop}, gens={args.gens}, "
          f"sigma={args.sigma}, 代理样本={args.eval_max_samples}）…", flush=True)

    try:
        import cma
    except ImportError:
        raise SystemExit("需要先 pip install cma")

    es = cma.CMAEvolutionStrategy(
        [1.0] * len(paths), args.sigma,
        {"bounds": [args.lo, args.hi], "popsize": args.pop,
         "verbose": -9, "seed": args.seed},
    )

    gen = 0
    while not es.stop() and gen < args.gens:
        gen += 1

        # es.ask()返回pop大小个候选
        xs = es.ask()
        print(f"  第 {gen} 代（{len(xs)} 个候选）", flush=True)
        fs = []
        for x in xs:
            t = time.time()
            score, accs = run_eval(list(x), args.eval_max_samples)
            fs.append(-float(score))
            record(x, score, accs, f"gen{gen}")
            print(f"    ({time.time() - t:.1f}s)", flush=True)
        es.tell(xs, fs)
        """
        es.tell()内部
        1. 按 fs 排名
        2. 改 m
        3. 改 σ
        4. 改 C
        5. 下一代由 (m,σ,C) 重新采样
        """

    # 取前topk个
    ranked = sorted(history, key=lambda h: -h["score"])
    cands, seen = [], set()
    for h in ranked:
        key = tuple(h["lams"])
        if key in seen:
            continue
        seen.add(key)
        cands.append(h)
        if len(cands) >= args.topk:
            break
    
    # 默认只有300个，这里用全量去验证候选
    print(f"\n全量复核 top-{len(cands)} …", flush=True)
    for h in cands:
        t = time.time()
        fs, fa = run_eval(h["lams"], 0)              # 0 = 全量
        h["full_score"] = round(float(fs), 5)
        h["full_accs"] = {k: round(float(v), 5) for k, v in fa.items()}
        print(f"  {h['lams']} → full={fs:.4f}（{time.time() - t:.1f}s）", flush=True)

    best = max(cands, key=lambda h: h["full_score"])
    print(f"\n最优 λ = {best['lams']}")
    print(f"  代理 {best['score']:.4f} → 全量 {best['full_score']:.4f}")
    for k, v in best["full_accs"].items():
        print(f"    {k:6s} {v:.4f}")

    if args.out:
        merger.apply(best["lams"])
        os.makedirs(args.out, exist_ok=True)
        merger.model.save_pretrained(args.out)
        merger.tokenizer.save_pretrained(args.out)
        print(f"\n模型 → {args.out}", flush=True)

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "history": history, "best": best},
                      f, ensure_ascii=False, indent=2)
        print(f"日志 → {args.log}", flush=True)


if __name__ == "__main__":
    main()
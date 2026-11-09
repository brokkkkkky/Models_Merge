"""DARE + Task Arithmetic：先随机丢弃增量，再按 λ 加权求和。

    theta_merged = theta0 + sum_i lam_i * DARE(delta_i)

和 task_arithmetic.py 的唯一区别：每个 delta 在累加前先过一遍 DARE
（随机丢掉 p 比例，剩下的放大 1/(1-p)）。

用法：
    # LoRA adapter（1.5B），λ=0.4，丢 10%
    python merge_method/dare_ta_merging.py --base Qwen/Qwen2.5-1.5B `
        --adapters outputs/mnli/adapter,outputs/qnli/adapter,... `
        --lam 0.4 --p 0.1 --seed 0 --out outputs/merged/ta_dare_p10_lam04

    # 全参 checkpoint（0.5B）
    python merge_method/dare_ta_merging.py --base Qwen/Qwen2.5-0.5B `
        --adapters outputs_ft/mnli/model,outputs_ft/qnli/model,... `
        --lam 0.4 --p 0.1 --seed 0 --out outputs/merged_ft/ta_dare_p10_lam04

λ 要和 p 匹配：DARE 把每个坐标放大 1/(1-p)，所以和普通 TA 等幅值的点是
    lam_dare = lam_ta * (1 - p)
例如普通 TA 最优 λ=0.4，则 p=0.1 对应 0.36，p=0.3 对应 0.28，p=0.5 对应 0.20。

输入目录两种都支持（open_source 按特征文件自动识别）：
    <dir>/adapter_model.safetensors  → LoRA adapter，delta = B @ A * (alpha / r)
    <dir>/model.safetensors          → 全参 checkpoint，delta = W_ft - W_base
tie_word_embeddings=True 时 lm_head 会被去重，取交集后自动跳过。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_arithmetic import AdapterDelta, open_source


def dare(delta: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    """随机丢弃 p 比例的增量，剩下的按 1/(1-p) 放大。"""
    if p <= 0.0:
        return delta
    mask = torch.rand(delta.shape, generator=gen) >= p
    return delta * mask / (1.0 - p)


def parse_args():
    p = argparse.ArgumentParser(description="DARE + Task Arithmetic")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapters", required=True,
                   help="逗号分隔；LoRA 指向 adapter，全参指向 model")
    p.add_argument("--lam", type=float, default=None,
                   help="统一权重，不填则默认 1/n（和普通 TA 的默认一致）")
    p.add_argument("--lams", default=None,
                   help="逗号分隔的逐模型权重，填了就覆盖 --lam")
    p.add_argument("--p", type=float, default=0.0, help="DARE 丢弃比例，0=关闭")
    p.add_argument("--seed", type=int, default=0, help="DARE 随机种子")
    p.add_argument("--skip_embed", action="store_true",
                   help="跳过 model.embed_tokens / lm_head，全参合并时省内存")
    p.add_argument("--max_modules", type=int, default=0, help="调试用，0=全部")
    p.add_argument("--dry_run", action="store_true", help="只算不存")
    p.add_argument("--out", default=None)
    return p.parse_args()


def resolve_lams(args, n: int) -> list[float]:
    if args.lams:
        lams = [float(s) for s in args.lams.split(",") if s.strip()]
        if len(lams) != n:
            raise ValueError(f"--lams 有 {len(lams)} 个，--adapters 有 {n} 个，必须一样多")
        return lams
    if args.lam is not None:
        return [args.lam] * n
    return [1.0 / n] * n


def main() -> None:
    args = parse_args()

    paths = [s.strip() for s in args.adapters.split(",") if s.strip()]
    if not paths:
        raise ValueError("--adapters 不能为空")

    lams = resolve_lams(args, len(paths))
    sources = [open_source(p) for p in paths]

    base_set = sources[0].names
    for s in sources[1:]:
        if s.names != base_set:
            raise ValueError(f"{s.path} 的模块集合与第一个不一致")

    modules = sorted(base_set)
    if args.skip_embed:
        modules = [m for m in modules if m not in ("model.embed_tokens", "lm_head")]
    if args.max_modules:
        modules = modules[: args.max_modules]

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    named = dict(model.named_modules())

    # LoRA 必须完全对得上；全参 checkpoint 里被去重的 lm_head 可以跳过
    missing = [m for m in modules if m not in named]
    if missing:
        if isinstance(sources[0], AdapterDelta):
            raise ValueError(f"模型模块缺失: {missing[:5]}")
        print(f"[提示] 跳过 {len(missing)} 个底座里没有的模块，例如 {missing[:3]}", flush=True)
        modules = [m for m in modules if m in named]

    print(f"DARE+TA: {len(paths)} 个模型, p={args.p}, lams={lams}, seed={args.seed}",
          flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for name in modules:
            w = named[name].weight
            total = None
            for src, lam in zip(sources, lams):
                d = dare(src.delta(name, w), args.p, gen)
                total = d * lam if total is None else total + d * lam
            w.add_(total.to(w.device, w.dtype))

    if args.dry_run:
        print("dry_run:未保存", flush=True)
        return

    out = args.out or "outputs/merged/dare_ta"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"完成 → {out}（合并 {len(modules)} 个模块）", flush=True)


if __name__ == "__main__":
    main()

"""TIES-Merging: Trim → Elect Sign → Disjoint Merge。"""

from __future__ import annotations
import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# load_adapter 和 load_scale 已经在 task_arithmetic.py 中实现了，直接导入即可。
from task_arithmetic import AdapterDelta, open_source

def keep_top_k(delta:torch.Tensor, keep:float) -> torch.Tensor:
    """只保留绝对值最大的 keep 比例，其余置零。"""
    if keep >= 1.0:
        return delta.clone()

    flat = delta.abs().flatten()

    # kthvalue 是 O(N) 选择算法，比 quantile 的排序快 3-5 倍，且精确
    # j = N×(1−keep) 是升序第 j 小的数，正好是"保留 keep 比例"的分界点
    n = flat.numel()
    k = max(1, int(round(n * keep)))       # 保留 k 个
    j = n - k + 1                          # 第 k 大的名次（升序）

    threshold = torch.kthvalue(flat, j).values

    # 绝对值不小于阈值的保留原值 ，否则置零
    return torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))


"""TIES-Merging 实现"""
def ties_merge(deltas:list[torch.Tensor], keep:float = 0.2) -> torch.Tensor:
    trimmed = [keep_top_k(d, keep) for d in deltas]
    stack = torch.stack(trimmed)

    # 每个矩阵每个参数位置各自独立决定方向
    elected = torch.sign(stack.sum(dim = 0))

    # 给出矩阵组每个元素的符号
    signs = torch.sign(stack)

    # 只保留与多数方向一致的 delta，其他置零
    # agree为 bool 张量
    agree = (signs == elected) & (stack != 0)

    count = agree.sum(dim = 0)
    summed = (stack * agree).sum(dim=0)

    # 求和取平均
    return summed / count.clamp(min=1) 

""" 命令行参数 """
def parse_args():
    p = argparse.ArgumentParser(description="TIES-Merging")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapters", required=True, help="逗号分隔")
    p.add_argument("--lam", type=float, default=1.0, help="默认 1.0，不要除以 n")
    p.add_argument("--lams", default=None,
                   help="逗号分隔的逐模型权重（长度=模型数），填了就覆盖 --lam。"
                        "权重作用在 TIES 之前，所以可以给不同任务不同的力度")
    p.add_argument("--trim", type=float, default=0.2)
    p.add_argument("--skip_embed", action="store_true",
                   help="跳过 model.embed_tokens / lm_head，全参合并时省内存")
    p.add_argument("--max_modules", type=int, default=0, help="调试用，0=全部")
    p.add_argument("--dry_run", action="store_true", help="只算不存")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 分解adapters参数
    paths = []
    for s in args.adapters.split(","):
        if s.strip():
            paths.append(s.strip())

    # 这里用 open_source 打开，groups里面是类：FullFTDelta 或者 AdapterDelta
    groups = [open_source(p) for p in paths]

    # 有lams就用lams，没有就用lam，长度必须和groups一样
    # TIES 是正齐次的，均匀权重放在前/后等价；非均匀必须在前面，否则没有意义
    if args.lams:
        lams = [float(s) for s in args.lams.split(",") if s.strip()]

        # 判断 lam 是否与 groups 数量一致
        if len(lams) != len(groups):
            raise ValueError(f"--lams 有 {len(lams)} 个，--adapters 有 {len(groups)} 个，必须一样多")
    else:
        lams = [args.lam] * len(groups)

    base_set = groups[0].names
    # 判断模块是否一致
    for g in groups[1:]:
        if g.names != base_set:
            raise ValueError(f"{g.path} 的模块集合与第一个不一致")

    modules = sorted(base_set)

    # 调试的时候使用
    if args.skip_embed:
        modules = [m for m in modules if m not in ("model.embed_tokens", "lm_head")]
    if args.max_modules:
        modules = modules[:args.max_modules]

    # 加载初始模型Qwen/Qwen2.5-1.5B，加载分词器
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype = torch.bfloat16, device_map = {"":0}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    named = dict(model.named_modules())

    # 检查是否缺失：LoRA 必须完全对得上；全参 checkpoint 里被去重的 lm_head 可以跳过
    missing = [n for n in modules if n not in named]
    if missing:
        if isinstance(groups[0], AdapterDelta):
            raise ValueError(f"模型模块缺失: {missing[:5]}")
        print(f"[提示] 跳过 {len(missing)} 个底座里没有的模块，例如 {missing[:3]}", flush=True)
        modules = [n for n in modules if n in named]

    with torch.no_grad():
        # 对每一层的不同矩阵使用TIES MERGING
        for idx, name in enumerate(modules,1):
            # 逐任务加权放在 TIES 之前：会影响符号选举和裁剪，这正是我们要搜的东西
            deltas = [lam * g.delta(name, named[name].weight)
                      for lam, g in zip(lams, groups)]

            # 调用TIES 计算τ
            tau = ties_merge(deltas, keep = args.trim)

            w = named[name].weight
            w.add_(tau.to(w.device, w.dtype))

    if args.dry_run:
        print("dry_run:未保存", flush = True)
        return

    out = args.out or "outputs/merged/ties"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"完成 → {out}", flush=True)

if __name__ == "__main__":
    main()

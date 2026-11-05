from __future__ import annotations

""" DARE + TIES-Merging: 随机丢弃 + Trim → Elect Sign → Disjoint Merge。"""
"""
附录 A.4 的标题是："Details of Grid Search on Hyperparameters of Model Merging Methods for Encoder-based LMs"——他们对 encoder 模型（GLUE）的合并超参做了网格搜索。
也就是说，论文里 GLUE 那张表的数字，每个方法(包括 DARE+TIES)的 λ 都是搜过的，不是固定在 1。
"""


import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# load_adapter 和 load_scale 已经在 task_arithmetic.py 中实现了，直接导入即可。
from task_arithmetic import load_adapter, load_scale

# 对于dare + ties 这里需要调整
def keep_top_k(delta:torch.Tensor, keep:float) -> torch.Tensor:
    """只保留绝对值最大的 keep 比例，其余置零。"""
    if keep >= 1.0:
        return delta.clone()


    flat = delta.abs().flatten()

    # 选取非0元素
    flat_nz = flat[flat > 0]

    # kthvalue 是 O(N) 选择算法，比 quantile 的排序快 3-5 倍，且精确
    # j = N×(1−keep) 是升序第 j 小的数，正好是"保留 keep 比例"的分界点
    n = flat_nz.numel()
    k = max(1, int(round(n * keep)))       # 保留 k 个
    j = n - k + 1                          # 第 k 大的名次（升序）

    threshold = torch.kthvalue(flat_nz, j).values

    # 绝对值不小于阈值的保留原值 ，否则置零
    return torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))


# DARE 实现
def dare(delta:torch.Tensor, p:float, gen:torch.Generator) -> torch.Tensor:
    """DARE: 随机丢弃 p 比例的增量，剩下的按 1/(1-p) 放大。"""
    if p <= 0.0:
        return delta
    
    mask = torch.rand(delta.shape, generator = gen) >= p
    return delta * mask / (1.0 - p)



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
    p.add_argument("--trim", type=float, default=0.2)
    p.add_argument("--max_modules", type=int, default=0, help="调试用，0=全部")
    p.add_argument("--dry_run", action="store_true", help="只算不存")
    p.add_argument("--out", default=None)

    # DARE 新增
    p.add_argument("--p", type=float, default=0.0, help="DARE 丢弃比例，0=关闭")
    p.add_argument("--seed", type=int, default=0, help="DARE 随机种子")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 分解adapters参数
    paths = []
    for s in args.adapters.split(","):
        if s.strip():
            paths.append(s.strip())

    groups = [
        {"path":p, "scale":load_scale(p), "tensors":load_adapter(p)} for p in paths
    ]

    base_set = set(groups[0]["tensors"])
    # 判断模块是否一致
    for g in groups[1:]:
        if set(g["tensors"]) != base_set:
            raise ValueError("模块集合不一致")

    modules = sorted(base_set)
    if args.max_modules:
        modules = modules[:args.max_modules]

    # 加载初始模型Qwen/Qwen2.5-1.5B
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype = torch.bfloat16, device_map = {"":0}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    named = dict(model.named_modules())

    # 检查是否缺失
    missing = []
    for n in modules:
        if n not in named:
            missing.append(n)

    if missing:
        raise ValueError("模型模块缺失")

    # 掩码生成的规则
    gen = torch.Generator().manual_seed(args.seed)

    with torch.no_grad():
        # 对每一层的不同矩阵使用TIES MERGING
        for idx, name in enumerate(modules,1):
            deltas = []
            for g in groups:
                # LORA的 A,B 矩阵
                pair = g["tensors"][name]
                #deltas.append((pair["B"].float() @ pair["A"].float()) * g["scale"])

                # 引入 DARE 的写法
                d = (pair["B"].float() @ pair["A"].float()) * g["scale"]
                deltas.append(dare(d, args.p, gen))

            # 调用TIES 计算τ
            tau = ties_merge(deltas, keep = args.trim)

            w = named[name].weight
            w.add_(tau.to(w.device, w.dtype) * args.lam)

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

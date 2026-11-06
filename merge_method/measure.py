"""DARE 之前先做的体检：看每个任务的 delta 量级分布和稠密度。

    python merge_method/measure.py                                   # LoRA adapter
    python merge_method/measure.py --full_ft --base Qwen/Qwen2.5-0.5B  # 全参 checkpoint

判断标准（DARE 的前提）：如果 delta 绝大多数都远小于 0.002 且非常稠密，
说明里面有大量可直接丢弃的冗余，DARE 才有发挥空间；LoRA 的低秩 delta
通常不满足这一点。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from task_arithmetic import open_source

NAMES = ["sst2", "cola", "mnli", "qnli", "rte", "qqp", "mrpc"]


def parse_args():
    p = argparse.ArgumentParser(description="delta 量级体检")
    p.add_argument("--full_ft", action="store_true", help="输入是全参微调 checkpoint")
    p.add_argument("--root", default=None, help="默认 outputs/ 或 outputs_ft/")
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B", help="全参模式下用来减底座")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root or ("outputs_ft" if args.full_ft else "outputs")
    sub = "model" if args.full_ft else "adapter"

    base_sd = None
    if args.full_ft:
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
        base_sd = m.state_dict()

    print(f"{'task':8s} {'vmax':>9s} {'p99.9':>9s} {'p99':>9s} {'mean':>9s} {'nonzero':>9s}")
    for name in NAMES:
        path = os.path.join(root, name, sub)
        if not os.path.isdir(path):
            print(f"{name:8s}    (missing)")
            continue

        src = open_source(path)
        samples = []
        vmax = 0.0
        nz = 0
        tot = 0
        for mod in sorted(src.names):
            bw = base_sd[mod + ".weight"] if base_sd is not None else None
            d = src.delta(mod, bw)
            a = d.abs().flatten()
            vmax = max(vmax, float(a.max()))
            step = max(1, a.numel() // 50_000)     # 每个模块抽 5 万个样本
            samples.append(a[::step])
            nz += int((d != 0).sum())
            tot += d.numel()

        v = torch.cat(samples)
        p999 = float(torch.quantile(v, 0.999))      # 99.9% 的元素小于这个值
        p99 = float(torch.quantile(v, 0.99))        # 99%   的元素小于这个值
        mean = float(v.mean())
        print(f"{name:8s} {vmax:9.5f} {p999:9.5f} {p99:9.5f} {mean:9.5f} "
              f"{100.0 * nz / max(tot, 1):8.2f}%")


if __name__ == "__main__":
    main()

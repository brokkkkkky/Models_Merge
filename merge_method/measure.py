"""
使用 DARE 之前运行，判断delta参数的值是否大多数小于 0.002
"""

import sys
sys.path.insert(0, "merge_method")
import torch
from task_arithmetic import load_adapter, load_scale

NAMES = ["sst2", "cola", "mnli", "qnli", "rte", "qqp", "mrpc"]


for name in NAMES:
    path = f"outputs/{name}/adapter"
    g = load_adapter(path)
    s = load_scale(path)

    samples = []
    vmax = 0.0
    for t in g.values():
        d = ((t["B"].float() @ t["A"].float()) * s).abs().flatten()
        vmax = max(vmax, float(d.max()))
        step = max(1, d.numel() // 50_000)     # 每个模块抽 5 万个样本
        samples.append(d[::step])

    v = torch.cat(samples)

    p999 = float(torch.quantile(v, 0.999)) # 99.9% 的元素小于这个值
    p99 = float(torch.quantile(v, 0.99))   # 99%   的元素小于这个值
    mean = float(v.mean())                 # 平均值
    print(f"{name:8s} {vmax:9.5f} {p999:9.5f} {p99:9.5f} {mean:9.5f} ")
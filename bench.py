"""定位训练慢在哪：分别测「纯模型」和「数据管道」的耗时。

用法：
    python bench.py
    python bench.py --bs 8 --seq 64 --no_grad_ckpt

判断标准（RTX 5060 Ti 8G 上）：
    纯模型 fwd+bwd   0.05–0.2 秒/步   正常
                     2 秒以上          环境有问题（驱动/内核/WDDM）
    数据管道         0.001–0.01 秒/批  正常
                     0.5 秒以上        瓶颈在取数据，开 num_workers
"""

from __future__ import annotations

import argparse
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--no_grad_ckpt", action="store_true", help="关闭 gradient checkpointing")
    return p.parse_args()


def timeit(fn, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> None:
    args = parse_args()
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"device {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    print(f"加载模型: {time.time() - t0:.1f} 秒")

    model.config.use_cache = False
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
    ))
    print(f"gradient checkpointing: {'关闭' if args.no_grad_ckpt else '开启'}")

    # 模拟真实样本：只有最后 2 个位置算 loss
    ids = torch.randint(0, 1000, (args.bs, args.seq), device="cuda")
    labels = ids.clone()
    labels[:, : args.seq - 2] = -100

    def model_step():
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    model_step()  # 先跑一次，触发显存分配
    per_step = timeit(model_step, args.iters)
    print(f"\n纯模型 fwd+bwd : {per_step:.3f} 秒/步   "
          f"显存 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    # ---- 数据管道 ----
    from finetune import CausalCollator  # 复用训练时同一个 collator

    collator = CausalCollator(1000)
    fake = [
        {"input_ids": list(range(10, 10 + args.seq)), "labels": [-100] * (args.seq - 2) + [1, 2]}
        for _ in range(256)
    ]
    idx = list(range(len(fake)))

    def data_step():
        batch = collator([fake[i] for i in idx[: args.bs]])
        return {k: v.to("cuda") for k, v in batch.items()}

    per_batch = timeit(data_step, 50)
    print(f"数据 + collate  : {per_batch * 1000:.2f} 毫秒/批")

    total = per_step + per_batch
    print(f"\n预计单步合计   : {total:.3f} 秒")
    print(f"4210 步约需    : {total * 4210 / 60:.0f} 分钟")

    if per_step > 1.0:
        print("\n>> 纯模型就慢，问题在 GPU 环境层（驱动/WDDM/内核），不在数据管道")
    elif per_batch > 0.05:
        print("\n>> 模型很快，瓶颈在数据管道：试试 --num_workers 2")
    else:
        print("\n>> 都正常，之前的慢可能来自显存不足换页或其它进程干扰")


if __name__ == "__main__":
    main()

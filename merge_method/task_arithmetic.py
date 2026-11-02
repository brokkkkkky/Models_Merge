import argparse
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def load_scale(adapter_dir: str) -> float:
    """读取 adapter_config.json 中的 alpha / r，计算 scale"""

    with open(os.path.join(adapter_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["lora_alpha"] / config["r"]


def load_adapter(adapter_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    """整理成 {模块名: {"A": tensor, "B": tensor}}，应为 196 个模块。"""
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    grouped: dict[str, dict[str, torch.Tensor]] = {}

    for key, tensor in sd.items():
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"Unexpected key {key} in adapter model.")
        if key.endswith(LORA_A_SUFFIX):
            name = key[len(ADAPTER_PREFIX):-len(LORA_A_SUFFIX)]
            grouped.setdefault(name, {})["A"] = tensor
        elif key.endswith(LORA_B_SUFFIX):
            name = key[len(ADAPTER_PREFIX):-len(LORA_B_SUFFIX)]
            grouped.setdefault(name, {})["B"] = tensor

    for name, pair in grouped.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing A or B for module {name}.")

    return grouped


def merge(base, a1_dir, a2_dir, lam1, lam2, out) -> None:
    s1, s2 = load_scale(a1_dir), load_scale(a2_dir)
    a1, a2 = load_adapter(a1_dir), load_adapter(a2_dir)

    print(f"scale1={s1}  scale2={s2}  模块数={len(a1)}/{len(a2)}", flush=True)
    if set(a1) != set(a2):
        raise ValueError("两个 adapter 模块不一致。")
    if len(a1) != 196:
        print(f"[警告] 模块数 {len(a1)}，预期 196", flush=True)

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(base)

    # named 的格式：{模块路径名: 模块对象}
    #   "": Qwen2ForCausalLM
    #   "model": Qwen2Model
    #   "model.embed_tokens": Embedding
    #   "model.layers": ModuleList
    #   "model.layers.0": Qwen2DecoderLayer
    #   "model.layers.0.self_attn": Qwen2Attention
    #   "model.layers.0.self_attn.q_proj": Linear          ← 我们要的
    #   "model.layers.0.mlp.gate_proj": Linear
    #   ...
    #   "lm_head": Linear

    named = dict(model.named_modules())

    missing = []
    for n in a1:
        if n not in named:
            missing.append(n)

    if missing:
        raise ValueError(f"以下模块在 base_model 中未找到: {missing}")

    names = sorted(a1)
    with torch.no_grad():
        for name in names:
            w = named[name].weight
            d1 = (a1[name]["B"].float() @ a1[name]["A"].float()) * s1  # delta 1
            d2 = (a2[name]["B"].float() @ a2[name]["A"].float()) * s2  # delta 2
            w.add_((lam1 * d1 + lam2 * d2).to(w.device, w.dtype))

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"完成 → {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="任务算术: theta0 + lam1*tau1 + lam2*tau2")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--a1", default="outputs/sst2/adapter")
    p.add_argument("--a2", default="outputs/agnews/adapter")
    p.add_argument("--lam1", type=float, default=0.5)
    p.add_argument("--lam2", type=float, default=0.5)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or f"outputs/merged/lam{args.lam1}_{args.lam2}"
    print(f"λ1={args.lam1} λ2={args.lam2} → {out}", flush=True)

    merge(args.base, args.a1, args.a2, args.lam1, args.lam2, out)


if __name__ == "__main__":
    main()
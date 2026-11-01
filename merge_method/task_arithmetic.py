import json                              
import os                                
import torch                             
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file  

ADAPTER_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

def load_scale(adapter_dir:str)->float:
    """读取 adapter_config.json 中的 alpha / r，计算 scale"""

    with open(os.path.join(adapter_dir,"adapter_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["lora_alpha"] / config["r"]


def load_adapter(adpter_dir:str)->dict[str,dict[str,torch.Tensor]]:
    """整理成 {模块名: {"A": tensor, "B": tensor}}，应为 196 个模块。"""
    sd = load_file(os.path.join(adpter_dir, "adapter_model.safetensors"))
    grouped: dict[str, dict[str, torch.Tensor]] = {}

    for key,tensor in sd.items():
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"Unexpected key {key} in adapter model.")
        if key.endswith(LORA_A_SUFFIX):
            name = key[len(ADAPTER_PREFIX):-len(LORA_A_SUFFIX)]
            grouped.setdefault(name, {})["A"] = tensor
        elif key.endswith(LORA_B_SUFFIX):
            name = key[len(ADAPTER_PREFIX):-len(LORA_B_SUFFIX)]
            grouped.setdefault(name, {})["B"] = tensor

    for name, group in grouped.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing A or B for module {name}.")

    return grouped


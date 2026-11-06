import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

DEFAULT_ADAPTERS = "outputs/sst2/adapter,outputs/agnews/adapter"


def load_config(adapter_dir: str) -> dict:
    with open(os.path.join(adapter_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_scale(adapter_dir: str) -> float:
    """读取 adapter_config.json 中的 alpha / r，计算 scale"""
    config = load_config(adapter_dir)
    return config["lora_alpha"] / config["r"]


def declared_base(adapter_dir: str) -> str | None:
    """adapter 训练时用的底座名字，可能为空。"""
    return load_config(adapter_dir).get("base_model_name_or_path")


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


WEIGHT_SUFFIX = ".weight"


class AdapterDelta:
    """
    LoRA adapter：delta = B @ A * (alpha / r)。

    这段计算原来内联在 merge() 里，搬到类里是为了让 ties/dare 脚本和全参
    checkpoint 共用同一套接口，数值上与原实现逐位一致。
    """

    def __init__(self, path: str):
        self.path = path
        self.scale = load_scale(path)
        self._t = load_adapter(path)
        self.names = set(self._t)

    def delta(self, name: str, base_weight: torch.Tensor) -> torch.Tensor:
        p = self._t[name]
        return (p["B"].float() @ p["A"].float()) * self.scale


class FullFTDelta:
    """
    全参微调 checkpoint：delta = W_ft - W_base。

    safe_open 是内存映射，按张量读取，7 个 checkpoint 不会同时整块进内存。
    注意 Qwen2.5 的 tie_word_embeddings=true 会让 save_pretrained 去重
    lm_head.weight，所以用 base_set & set(named) 取交集即可自动跳过它。
    """

    def __init__(self, path: str):
        self.path = path
        sd = os.path.join(path, "model.safetensors")

        # 判断是否存在
        if not os.path.exists(sd):
            if os.path.exists(os.path.join(path, "model.safetensors.index.json")):
                raise ValueError(f"{path} 是分片 checkpoint，本实现不支持")
            raise FileNotFoundError(f"找不到 {sd}")
        
        self._h = safe_open(sd, framework="pt", device="cpu")
        self.names = {
            k[: -len(WEIGHT_SUFFIX)] for k in self._h.keys() if k.endswith(WEIGHT_SUFFIX)
        }
    
    # 计算任务向量
    def delta(self, name: str, base_weight: torch.Tensor) -> torch.Tensor:
        # 一律在 CPU 上算：embed_tokens 这类大模块的 fp32 delta 有 500 MB 以上，
        # 7 份同时留在显存里会 OOM
        w = self._h.get_tensor(name + WEIGHT_SUFFIX).to(torch.float32)
        base = base_weight.detach().to(device="cpu", dtype=torch.float32)
        return w - base


def open_source(path: str):
    """
    按目录里的特征文件识别 delta 来源：
    1. <dir>/adapter_model.safetensors → LoRA adapter
    2. <dir>/model.safetensors         → 全参微调 checkpoint
    """

    if os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        return AdapterDelta(path)
    if os.path.exists(os.path.join(path, "model.safetensors")):
        return FullFTDelta(path)
    raise FileNotFoundError(
        f"{path} 下既没有 adapter_model.safetensors 也没有 model.safetensors。"
        f"LoRA 请指向 outputs/<task>/adapter，全参请指向 outputs_ft/<task>/model"
    )

# Lora微调使用
def merge(base, specs, out) -> None:
    """specs: [(adapter_dir, lam), ...]，任意个，lam 可以为负。"""

    # lam == 0 的 adapter 不用加载，直接跳过
    # 只加载 lam != 0 的 adapter，避免浪费内存
    used = [(path, lam) for path, lam in specs if lam != 0.0]
    if not used:
        raise ValueError("所有 lam 都是 0，没有可合并的内容。")

    groups = []
    for path, lam in used:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"adapter 目录不存在: {path}")
        groups.append(
            {"path": path, "lam": lam, "scale": load_scale(path), "tensors": load_adapter(path)}
        )
    
    #groups的格式
    #groups = [
    #{                                        # ← 第一个 adapter
    #    "path":    "outputs/sst2/adapter",   # str
    #    "lam":     0.5,                      # float
    #    "scale":   2.0,                      # float
    #    "tensors": {                         # dict，196 个模块
    #        "model.layers.0.self_attn.q_proj": {
    #            "A": Tensor(16, 1536),
    #            "B": Tensor(1536, 16),
    #        },
    #        "model.layers.0.self_attn.k_proj": {
    #            "A": Tensor(16, 1536),
    #            "B": Tensor(256, 16),
    #        },
    #        ...
    #    },
    #},
    #]

    
    # 底座一致性：各 adapter 自己声明的底座必须相同，否则合并出来的东西没有意义
    declared = {d for d in (declared_base(g["path"]) for g in groups) if d}
    if len(declared) > 1:
        raise ValueError(f"这些 adapter 声明的底座不一致: {sorted(declared)}")
    if declared and base not in declared:
        print(f"[警告] adapter 声明的底座是 {next(iter(declared))}，当前 --base 是 {base}",
              flush=True)

    # 模块集合一致性：各 adapter 的模块集合必须相同，否则合并出来的东西没有意义
    # 每个微调后模型的base_set都一样
    base_set = set(groups[0]["tensors"])
    for g in groups[1:]:
        if set(g["tensors"]) != base_set:
            diff = list(base_set ^ set(g["tensors"]))[:3]
            raise ValueError(f"{g['path']} 的模块集合和第一个 adapter 不一致，例如: {diff}")
    if len(base_set) != 196:
        print(f"[警告] 模块数 {len(base_set)}，预期 196", flush=True)


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

    missing = [n for n in base_set if n not in named]
    if missing:
        raise ValueError(f"以下模块在 base_model 中未找到: {missing[:5]}")

    names = sorted(base_set)
    with torch.no_grad():
        for name in names:
            w = named[name].weight  # 基础模型的权重
            total = None
            for g in groups:
                pair = g["tensors"][name]
                delta = (pair["B"].float() @ pair["A"].float()) * g["scale"]
                total = delta * g["lam"] if total is None else total + delta * g["lam"]
            w.add_(total.to(w.device, w.dtype))

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"完成 → {out}", flush=True)

# 全参数微调使用
def merge_sources(specs, out: str, base: str) -> None:
    """
    全参微调 checkpoint 的合并：theta0 + sum_i lam_i * (W_ft_i - W_base)。

    specs: [(checkpoint_dir, lam), ...]
    """

    # 判断哪些模型是需要使用的
    used = [(path, lam) for path, lam in specs if lam != 0.0]
    if not used:
        raise ValueError("所有 lam 都是 0，没有可合并的内容。")

    sources = [open_source(path) for path, _ in used]
    base_set = sources[0].names
    
    for s in sources[1:]:
        if s.names != base_set:
            raise ValueError(f"{s.path} 的模块集合与第一个不一致")

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(base)

    named = dict(model.named_modules())
    # 取交集：tie_word_embeddings=True 时 lm_head.weight 会被去重，不在 names 里，
    # 跳过它是正确的（更新 embed_tokens 等于更新 lm_head，不能加两次）
    names = sorted(base_set & set(named))
    skipped = sorted(base_set - set(named))
    if skipped:
        print(f"[提示] 跳过 {len(skipped)} 个 checkpoint 里有、模型里没有的模块，例如 {skipped[:3]}",
              flush=True)

    with torch.no_grad():
        for name in names:
            w = named[name].weight
            total = None
            for src, (_, lam) in zip(sources, used):
                d = src.delta(name, w)
                total = d * lam if total is None else total + d * lam
            w.add_(total.to(w.device, w.dtype))

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"完成 → {out}（合并 {len(names)} 个模块）", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="任务算术: theta0 + sum_i lam_i * tau_i")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument(
        "--adapters",
        default=DEFAULT_ADAPTERS,
        help="逗号分隔的目录，可以有 1 个或多个；LoRA 指向 adapter，全参指向 model",
    )
    p.add_argument(
        "--lams",
        default = None,
        help = "逗号分隔的权重，个数必须和 --adapters 一致，可为负数。不填则默认 1/n即对任务向量取平均",
    )
    p.add_argument("--full_ft", action="store_true",
                   help="输入是全参微调 checkpoint（<dir>/model.safetensors）")
    p.add_argument("--out", default=None)
    return p.parse_args()


def fmt_lam(lam: float) -> str:
    """把 0.3333333 这类数压成 0.333，用于目录名。"""
    s = f"{lam:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def main() -> None:
    args = parse_args()

    adapters = [s.strip() for s in args.adapters.split(",") if s.strip()]
    if not adapters:
        raise ValueError("--adapters 不能为空")

    if args.lams is None:
        # 默认 1/n：对任务向量取平均
        lams = [1.0 / len(adapters)] * len(adapters)
        auto = True
    else:
        lams = [float(s) for s in args.lams.split(",") if s.strip()]
        auto = False

    # 校验模型数是否等于权重数
    if len(adapters) != len(lams):
        raise ValueError(
            f"--adapters 有 {len(adapters)} 个，--lams 有 {len(lams)} 个，必须一样多"
        )

    # 全参合并落到单独的目录前缀，避免和 LoRA 的 avg_N / lam... 互相覆盖
    prefix = "outputs/merged_ft" if args.full_ft else "outputs/merged"
    if args.out:
        out = args.out
    elif auto:
        out = f"{prefix}/avg_{len(adapters)}"
    else:
        out = f"{prefix}/lam" + "_".join(fmt_lam(lam) for lam in lams)

    if args.full_ft:
        print(f"合并 {len(adapters)} 个全参 checkpoint，lam={lams} → {out}", flush=True)
        merge_sources(list(zip(adapters, lams)), out, args.base)
    else:
        print(f"合并 {len(adapters)} 个 adapter，lam={lams} → {out}", flush=True)
        merge(args.base, list(zip(adapters, lams)), out)


if __name__ == "__main__":
    main()

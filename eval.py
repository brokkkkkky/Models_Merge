"""统一评测入口：base / base+adapter / 融合后的完整 checkpoint。

三种用法：
    python eval.py --task sst2                                 # base，零样本（θ₀）
    python eval.py --task sst2 --adapter outputs/sst2/adapter  # 单任务模型
    python eval.py --task sst2 --base outputs/merged_lam1.0    # 融合后的 checkpoint

打分口径全部来自 common.py，脚本里不出现任何可调的 prompt 或标签词副本。
"""

from __future__ import annotations

import argparse
import os

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    TASKS,
    check_label_words,
    evaluate,
    get_task,
    load_split,
    print_report,
    save_json,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="任务算术统一评测")
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B", help="底座或已合并的 checkpoint")
    p.add_argument("--adapter", default=None, help="LoRA adapter 目录；给了就在 base 上挂载")
    p.add_argument("--split", default=None, help="不填则用任务默认验证集")
    p.add_argument("--max_samples", type=int, default=0, help="0 表示全量")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--data_path", default=None, help="本地 parquet 兜底路径")
    p.add_argument("--tag", default=None, help="结果文件名，默认按 base/adapter 自动取")
    p.add_argument("--out_dir", default="eval")
    p.add_argument("--show", type=int, default=5, help="打印几条样本")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_tag(args) -> str:
    if args.tag:
        return args.tag
    if args.adapter:
        return os.path.basename(os.path.normpath(args.adapter)) or args.task
    if args.base.rstrip("/\\") != "Qwen/Qwen2.5-1.5B":
        return os.path.basename(os.path.normpath(args.base))
    return "base"


def main() -> None:
    args = parse_args()
    spec = get_task(args.task)
    set_seed(args.seed)

    # 启动就打印环境状态，卡在下载/加载时能立刻判断原因
    print(
        f"[启动] task={spec.name} base={args.base} "
        f"HF_HOME={os.environ.get('HF_HOME', '<未设置>')} "
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<未设置>')}",
        flush=True,
    )
    print("[1/3] 加载 tokenizer …", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    check_label_words(tokenizer, spec)
    print(f"[1/3] tokenizer 就绪，标签词 {len(spec.label_words)} 个均为单 token", flush=True)
    print("[2/3] 加载模型 …", flush=True)

    # 注意：from_pretrained 默认把模型放在 CPU 上，必须显式指定 device_map，
    # 否则推理会跑在 CPU 上，慢到看起来像卡死
    if torch.cuda.is_available():
        device_map = {"": 0}
        print(f"[2/3] 使用 GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device_map = None
        print("[警告] 没有可用 GPU，将在 CPU 上推理，会非常慢", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    base_commit = getattr(model.config, "_commit_hash", None)

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"已挂载 adapter: {args.adapter}", flush=True)
    model.eval()
    print("[3/3] 开始评测 …", flush=True)

    dataset = load_split(spec, args.split, args.data_path)
    metrics = evaluate(
        model,
        tokenizer,
        spec,
        dataset,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_samples=args.max_samples,
        show=args.show,
    )
    print_report(metrics)

    tag = resolve_tag(args)
    path = os.path.join(args.out_dir, spec.name, f"{tag}.json")
    save_json(
        {
            **metrics,
            "tag": tag,
            "model": args.base,
            "adapter": args.adapter,
            "commit": base_commit,
            "env": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        },
        path,
    )
    print(f"\n结果 → {path}")


if __name__ == "__main__":
    main()

"""单任务 LoRA 微调：训练 → 验证集打分 → 存 adapter。

用法：
    python finetune.py --task sst2
    python finetune.py --task agnews

只保存 LoRA 的 A/B 矩阵（几十 MB），不合并回底座。合并留到需要时再做。
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import time

import peft
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import (
    TASKS,
    build_example,
    check_label_words,
    evaluate,
    get_task,
    load_split,
    print_report,
    save_json,
    set_seed,
)

# 两个任务的默认 epoch 数：让训练步数都落在 8000 上下。
# τ 的范数和训练步数正相关，步数差太多会让后续融合隐式加权。
DEFAULT_EPOCHS = {"sst2": 2, "agnews": 1, "cola": 3}

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class CausalCollator:
    """按 batch 内最长做动态 padding，labels 的 padding 位填 -100。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            ids = torch.tensor(f["input_ids"], dtype=torch.long)
            lab = torch.tensor(f["labels"], dtype=torch.long)
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
            labels[i, : len(lab)] = lab

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_rows(spec, tokenizer, split, max_len, max_samples, seed, data_path):
    """把整个 split 预 tokenize 进内存。返回 (rows, 被截断的条数)。"""
    ds = load_split(spec, split, data_path)
    if max_samples and max_samples > 0 and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    rows = []
    truncated = 0
    for ex in ds:
        ids, labels, trunc = build_example(
            tokenizer, spec, ex[spec.text_field], int(ex[spec.label_field]), max_len
        )
        truncated += int(trunc)
        rows.append({"input_ids": ids, "labels": labels})
    return rows, truncated


def parse_args():
    p = argparse.ArgumentParser(description="单任务 LoRA 微调")
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--bs", type=int, default=4, help="每卡 batch size")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=None, help="不填则用任务默认值")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_samples", type=int, default=0, help="0 表示用全量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_path", default=None, help="训练集本地 parquet 兜底路径")
    p.add_argument("--val_data_path", default=None, help="验证集本地 parquet；不填则复用 data_path")
    p.add_argument("--num_workers", type=int, default=0,
                   help="数据加载进程数。0 最稳；训练慢且 bench 显示瓶颈在数据时试 2-4")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="关闭 gradient checkpointing，快约 25%%，但多占约 1-2 GB 显存")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_task(args.task)
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[args.task]

    set_seed(args.seed)
    out_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)

    # ---- tokenizer 与标签词校验 ----
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_ids = check_label_words(tokenizer, spec)
    print("标签词 token id:", dict(zip(spec.label_words, label_ids)))

    # ---- 数据 ----
    rows, truncated = build_rows(
        spec, tokenizer, spec.train_split, args.max_len, args.max_samples, args.seed, args.data_path
    )
    print(f"训练样本 {len(rows)} 条，其中 {truncated} 条正文被截断")

    # ---- 模型 ----
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    base_commit = getattr(model.config, "_commit_hash", None)
    if base_commit is None:
        print("[警告] 拿不到 base 的 commit hash，meta.json 里会留空")

    model.config.use_cache = False
    # 顺序很重要：先开 checkpointing，再让输入 embed 参与求导，
    # 否则 LoRA 参数拿不到梯度，loss 完全不降。
    if args.no_grad_ckpt:
        print("gradient checkpointing: 关闭（用显存换速度）")
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 显式搬到显卡。Trainer 也会自己搬，但提前搬可以让显存问题在训练开始前暴露
    if torch.cuda.is_available():
        model.to("cuda")
        print(f"训练设备: {torch.cuda.get_device_name(0)}")
    else:
        print("[警告] 没有可用 GPU，训练会慢到无法接受")

    # transformers 5.x 移除了 warmup_ratio，这里自己按 3% 折算成步数
    steps_per_epoch = math.ceil(len(rows) / (args.bs * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup_steps = max(1, int(total_steps * 0.03))

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "run"),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.num_workers > 0,
        dataloader_persistent_workers=args.num_workers > 0,
        gradient_checkpointing=False,  # 已在模型上手动开启
        seed=args.seed,
    )
    print(f"总步数 {total_steps}，warmup {warmup_steps} 步")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )

    print("开始训练 …")
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    print(f"训练结束，用时 {train_minutes:.1f} 分钟")

    # ---- 验证集打分（adapter 还挂着，不用合并、不用重载）----
    val_ds = load_split(spec, spec.val_split, args.val_data_path or args.data_path)
    metrics = evaluate(
        model, tokenizer, spec, val_ds, batch_size=16, max_len=args.max_len
    )
    print_report(metrics)

    # ---- 保存 ----
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "task": spec.name,
        "base_model": args.base,
        "base_commit": base_commit,
        "prompt": {
            "prefix": spec.prefix,
            "suffix": spec.suffix,
            "label_words": list(spec.label_words),
        },
        "hyperparameters": {
            "max_len": args.max_len,
            "bs": args.bs,
            "grad_accum": args.grad_accum,
            "effective_batch": args.bs * args.grad_accum,
            "lr": args.lr,
            "epochs": epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": TARGET_MODULES,
            "seed": args.seed,
        },
        "train": {
            "split": spec.train_split,
            "samples": len(rows),
            "truncated": truncated,
            "minutes": round(train_minutes, 2),
        },
        "val": {
            "split": spec.val_split,
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "mean_nll": metrics["mean_nll"],
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_json(meta, os.path.join(out_dir, "meta.json"))

    eval_path = os.path.join("eval", spec.name, f"{spec.name}.json")
    save_json(
        {**metrics, "model": args.base, "adapter": adapter_dir, "commit": base_commit},
        eval_path,
    )

    print(f"\nadapter  → {adapter_dir}")
    print(f"meta     → {os.path.join(out_dir, 'meta.json')}")
    print(f"eval     → {eval_path}")


if __name__ == "__main__":
    main()

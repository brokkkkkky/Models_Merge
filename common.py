"""任务定义、prompt 构造与统一评测口径。

finetune.py 和 eval.py 都从这里取任务配置，保证训练和评测用的是同一个模板、
同一套标签词。融合脚本也应该复用这里的 evaluate()。
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import torch


# --------------------------------------------------------------------------
# 任务定义
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """一个任务需要的全部信息：数据在哪、怎么拼 prompt、答案词是什么。"""

    name: str
    dataset: str
    config: str | None
    train_split: str
    val_split: str
    text_field: str
    label_field: str
    prefix: str
    suffix: str
    label_words: tuple[str, ...]  # 顺序必须和数据集里的 label 整数一一对应


TASKS: dict[str, TaskSpec] = {
    "sst2": TaskSpec(
        name="sst2",
        dataset="nyu-mll/glue",
        config="sst2",
        train_split="train",
        val_split="validation",
        text_field="sentence",
        label_field="label",
        prefix="Review: ",
        # 用明确的问句：base model 看到孤零零的 "Sentiment:" 只会续写最顺口的词，
        # 实测会把 93% 的样本判成同一类（准确率 0.56）；改成问句后 0.945
        suffix="\nIs the sentiment positive or negative? Answer:",
        label_words=(" negative", " positive"),
    ),
    "agnews": TaskSpec(
        name="agnews",
        dataset="fancyzhx/ag_news",
        config=None,
        train_split="train",
        val_split="test",
        text_field="text",
        label_field="label",
        prefix="News: ",
        # 实测：写成完整的 "belongs to the category" 比光秃秃的 "Topic:" 好，
        # 且第 4 类必须用 " Technology" 而不是 " Tech"，否则该类几乎不被预测
        suffix="\nThis article belongs to the category",
        label_words=(" World", " Sports", " Business", " Technology"),
    ),
}


# 获取任务类型
def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"未知任务 {name!r}，可选：{list(TASKS)}")
    return TASKS[name]



# --------------------------------------------------------------------------
# prompt 构造
# --------------------------------------------------------------------------
def clean_text(text: Any) -> str:
    """压掉换行和连续空白，避免模板里出现不可控的空白差异。"""
    return " ".join(str(text).split())


def tokenize_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def check_label_words(tokenizer, spec: TaskSpec) -> list[int]:
    """确认每个标签词恰好是 1 个 token，否则直接报错。

    这是最容易静默出错的地方：标签词被切成两三个 token 时，取最后一个
    token 的 logit 依然能跑，但分数含义已经完全错了。
    """
    ids = []
    for word in spec.label_words:
        pieces = tokenize_no_special(tokenizer, word)
        if len(pieces) != 1:
            tokens = [tokenizer.convert_ids_to_tokens(i) for i in pieces]
            raise ValueError(
                f"[{spec.name}] 标签词 {word!r} 被切成 {len(pieces)} 个 token {tokens}，"
                f"请换成单 token 的写法"
            )
        ids.append(pieces[0])
    return ids

# 拼接 prompt
def build_prompt_ids(
    tokenizer,
    spec: TaskSpec,
    text: Any,
    max_len: int,
    answer_len: int = 2,
) -> tuple[list[int], bool]:
    """拼出 prompt 的 token id。

    prefix / 正文 / suffix 分段 tokenize 再拼接，训练和评测走同一条路径。
    超长时只截断正文那段，保证前缀和 "Topic:" 这类后缀完整。

    返回 (ids, 是否被截断)。
    """
    prefix_ids = tokenize_no_special(tokenizer, spec.prefix)
    suffix_ids = tokenize_no_special(tokenizer, spec.suffix)
    text_ids = tokenize_no_special(tokenizer, clean_text(text))

    budget = max_len - len(prefix_ids) - len(suffix_ids) - answer_len
    if budget < 8:
        raise ValueError(
            f"max_len={max_len} 太小：模板已占 {len(prefix_ids) + len(suffix_ids)} 个 token"
        )

    truncated = len(text_ids) > budget
    if truncated:
        text_ids = text_ids[:budget]

    return prefix_ids + text_ids + suffix_ids, truncated

# 训练/微调的时候用
def build_example(
    tokenizer,
    spec: TaskSpec,
    text: Any,
    label_idx: int,
    max_len: int,
) -> tuple[list[int], list[int], bool]:
    """构造一条训练样本：(input_ids, labels, 是否截断)。

    labels 的前缀部分是 -100，只在答案 token 和结尾 eos 上算 loss。
    """
    answer_ids = tokenize_no_special(tokenizer, spec.label_words[label_idx])
    if len(answer_ids) != 1:
        raise ValueError(f"[{spec.name}] 标签词 {spec.label_words[label_idx]!r} 不是单 token")

    prompt_ids, truncated = build_prompt_ids(tokenizer, spec, text, max_len, answer_len=2)

    # inputs_id作为训练时的输入，labels作为训练时的监督信号
    input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]

    # 只对两个位置计算损失
    labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
    return input_ids, labels, truncated


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_split(spec: TaskSpec, split: str | None = None, data_path: str | None = None):
    """从 Hub 或本地 parquet 读一个 split。

    data_path 用于 Hub 拉不动时的兜底：直接给一个 parquet 文件路径。
    """
    from datasets import load_dataset

    split = split or spec.val_split
    if data_path:
        return load_dataset("parquet", data_files=data_path, split="train")
    if spec.config:
        return load_dataset(spec.dataset, spec.config, split=split)
    return load_dataset(spec.dataset, split=split)




# --------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    spec: TaskSpec,
    dataset,
    batch_size: int = 16,
    max_len: int = 256,
    max_samples: int = 0,
    show: int = 5,
) -> dict[str, Any]:
    """用 verbalizer 口径给模型打分。

    只看最后一个 prompt token 之后、各标签词对应的 logit，取最大的那个。
    不生成、不比对字符串，所以结果稳定、可复现。
    """
    label_token_ids = check_label_words(tokenizer, spec)
    n_classes = len(label_token_ids)
    device = next(model.parameters()).device

    total = len(dataset) if max_samples in (None, 0) else min(max_samples, len(dataset))
    if total == 0:
        raise ValueError("评测集为空")

    model.eval()
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # 左 padding 才能用 logits[:, -1]

    correct = 0
    nll_sum = 0.0
    per_class_n = [0] * n_classes
    per_class_correct = [0] * n_classes
    confusion = [[0] * n_classes for _ in range(n_classes)]
    samples: list[dict[str, Any]] = []

    try:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            rows = dataset[start:end]
            texts = rows[spec.text_field]
            golds = [int(g) for g in rows[spec.label_field]]

            id_list = [build_prompt_ids(tokenizer, spec, t, max_len)[0] for t in texts]
            # 传 dict 列表而不是 {"input_ids": [...]}，这样 pad 才会自动补 attention_mask
            batch = tokenizer.pad(
                [{"input_ids": ids} for ids in id_list], padding=True, return_tensors="pt"
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits[:, -1, :].float()
            scores = logits[:, label_token_ids]  # (B, n_classes)
            logp = torch.log_softmax(scores, dim=-1)

            gold_t = torch.tensor(golds, device=scores.device)
            nll = -logp.gather(1, gold_t[:, None]).squeeze(1)
            nll_sum += float(nll.sum().item())

            preds = scores.argmax(dim=-1).tolist()
            for j, (pred, gold) in enumerate(zip(preds, golds)):
                per_class_n[gold] += 1
                confusion[gold][pred] += 1
                if pred == gold:
                    correct += 1
                    per_class_correct[gold] += 1
                if len(samples) < show:
                    # 用 batch 内的局部下标 j，不能拿全局 len(samples) 去索引 texts
                    samples.append(
                        {
                            "text": clean_text(texts[j])[:120],
                            "gold": spec.label_words[gold].strip(),
                            "pred": spec.label_words[pred].strip(),
                            "logits": {
                                spec.label_words[i].strip(): round(float(scores[j, i]), 3)
                                for i in range(n_classes)
                            },
                        }
                    )
    finally:
        tokenizer.padding_side = old_side

    result: dict[str, Any] = {
        "task": spec.name,
        "n": total,
        "accuracy": correct / total,
        "mean_nll": nll_sum / total,
        "per_class": {
            spec.label_words[i].strip(): {
                "n": per_class_n[i],
                "accuracy": (per_class_correct[i] / per_class_n[i]) if per_class_n[i] else None,
            }
            for i in range(n_classes)
        },
        "confusion_matrix": confusion,
        "label_words": [w.strip() for w in spec.label_words],
        "samples": samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def print_report(result: dict[str, Any]) -> None:
    """把评测结果打印。"""
    print(
        f"\n=== {result['task']}  n={result['n']}  "
        f"acc={result['accuracy']:.4f}  nll={result['mean_nll']:.4f} ==="
    )
    for word, cls in result["per_class"].items():
        acc = cls["accuracy"]
        acc_str = "n/a" if acc is None else f"{acc:.4f}"
        print(f"  {word:>10s}  n={cls['n']:<6d}  acc={acc_str}")

    labels = result["label_words"]
    if result["confusion_matrix"]:
        header = "  ".join(f"{w[:8]:>8s}" for w in labels)
        print(f"\n  混淆矩阵（行=真值，列=预测）\n  {'':>10s}  {header}")
        for i, row in enumerate(result["confusion_matrix"]):
            cells = "  ".join(f"{v:>8d}" for v in row)
            print(f"  {labels[i]:>10s}  {cells}")

    for s in result["samples"]:
        print(f"\n  [{s['gold']} -> {s['pred']}] {s['text']}")
        print(f"      logits: {s['logits']}")


# --------------------------------------------------------------------------
# 杂项
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def save_json(obj: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

"""对比不同 prompt 模板在同一任务上的零样本准确率。

base model 对模板很敏感：模板不明确时，模型只会续写"最顺口"的词，
而不是在两个标签之间做选择，表现为预测几乎全挤在同一类。

用法：
    python diagnose_template.py --task sst2
    python diagnose_template.py --task agnews --max_samples 2000
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import TASKS, clean_text, load_split, set_seed, tokenize_no_special

# 每个任务给几组候选：(名称, prefix, suffix, label_words)
# label_words 的顺序必须和数据集 label 整数一一对应
CANDIDATES = {
    "sst2": [
        (
            "T1 当前",
            "Review: ",
            "\nSentiment:",
            (" negative", " positive"),
        ),
        (
            "T2 问句",
            "Review: ",
            "\nIs the sentiment positive or negative? Answer:",
            (" negative", " positive"),
        ),
        (
            "T3 自然续写",
            "Movie review: ",
            "\nIn one word, this movie was",
            (" bad", " good"),
        ),
        (
            "T4 反应",
            "Review: ",
            "\nMy overall reaction was",
            (" negative", " positive"),
        ),
    ],
    "agnews": [
        (
            "A1 Topic:+Tech",
            "News: ",
            "\nTopic:",
            (" World", " Sports", " Business", " Tech"),
        ),
        (
            "A2 类别+Tech",
            "News: ",
            "\nThis article belongs to the category",
            (" World", " Sports", " Business", " Tech"),
        ),
        (
            "A3 类别+Technology",
            "News: ",
            "\nThis article belongs to the category",
            (" World", " Sports", " Business", " Technology"),
        ),
        (
            "A4 问句+Technology",
            "News: ",
            "\nWhich topic is this news about? World, Sports, Business, or Sci/Tech?\nAnswer:",
            (" World", " Sports", " Business", " Technology"),
        ),
    ],
    "cola": [
        (
            "C1 当前",
            "Sentence: ",
            "\nIs this sentence grammatically correct? Answer:",
            (" no", " yes"),
        ),
        (
            "C2 语法续写",
            "Sentence: ",
            "\nThis sentence is grammatically",
            (" incorrect", " correct"),
        ),
        (
            "C3 可接受性",
            "Sentence: ",
            "\nIs it grammatically acceptable? Answer:",
            (" no", " yes"),
        ),
        (
            "C4 明确指令",
            "Sentence: ",
            "\nDoes this sentence follow English grammar? Answer:",
            (" no", " yes"),
        ),
        (
            "C5 反向问错误",
            "Sentence: ",
            "\nIs there a grammar mistake in this sentence? Answer:",
            # 注意顺序：0=unacceptable（有错）→ yes；1=acceptable（没错）→ no
            (" yes", " no"),
        ),
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="对比 prompt 模板的零样本准确率")
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--split", default=None)
    p.add_argument("--max_samples", type=int, default=0, help="0 表示用全量")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--data_path", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def run_template(model, tokenizer, dataset, texts, golds, prefix, suffix, label_words,
                 batch_size, max_len):
    prefix_ids = tokenize_no_special(tokenizer, prefix)
    suffix_ids = tokenize_no_special(tokenizer, suffix)
    label_ids = [tokenize_no_special(tokenizer, w) for w in label_words]
    for w, ids in zip(label_words, label_ids):
        if len(ids) != 1:
            raise ValueError(f"标签词 {w!r} 不是单 token：{ids}")

    label_ids = [i[0] for i in label_ids]
    n_classes = len(label_ids)
    budget = max_len - len(prefix_ids) - len(suffix_ids)
    if budget < 8:
        raise ValueError(f"max_len={max_len} 太小")

    correct = 0
    dist = [0] * n_classes

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch = [
            prefix_ids
            + tokenize_no_special(tokenizer, clean_text(t))[:budget]
            + suffix_ids
            for t in chunk
        ]
        enc = tokenizer.pad([{"input_ids": b} for b in batch], padding=True,
                            return_tensors="pt").to(model.device)
        scores = model(**enc).logits[:, -1, :].float()[:, label_ids]
        for pred, gold in zip(scores.argmax(-1).tolist(), golds[start : start + batch_size]):
            dist[pred] += 1
            correct += int(pred == gold)

    return correct / len(texts), dist


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device_map
    ).eval()

    import common

    spec = common.get_task(args.task)
    dataset = load_split(spec, args.split, args.data_path)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    texts = list(dataset[spec.text_field])
    golds = [int(g) for g in dataset[spec.label_field]]
    print(f"任务 {spec.name}，样本 {len(texts)} 条\n")
    print(f"{'模板':<14s} {'准确率':>8s}   预测分布")
    print("-" * 60)

    results = []
    for name, prefix, suffix, words in CANDIDATES[args.task]:
        try:
            acc, dist = run_template(
                model, tokenizer, dataset, texts, golds,
                prefix, suffix, words, args.batch_size, args.max_len,
            )
        except ValueError as e:
            print(f"{name:<14s} 跳过：{e}")
            continue
        results.append((name, acc))
        print(f"{name:<14s} {acc:8.4f}   {dist}")

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n最佳：{best[0]}  acc={best[1]:.4f}")
        print("把对应的 prefix / suffix / label_words 写进 common.py 的 TASKS 即可。")


if __name__ == "__main__":
    main()

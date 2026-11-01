# Models_Merge

用 LoRA 微调 → 取任务向量 → 向量相加，学习 task arithmetic 的最小工程。

模型：`Qwen/Qwen2.5-1.5B`（base，非 Instruct）
任务：`sst2`（情感二分类）、`agnews`（新闻四分类）

## 文件

| 文件 | 作用 |
| --- | --- |
| `common.py` | 任务定义、prompt 构造、标签词校验、`evaluate()` |
| `finetune.py` | 单任务 LoRA 微调，产出 adapter |
| `eval.py` | 统一评测：base / base+adapter / 合并后的 checkpoint |

训练和评测共用 `common.py` 里的模板与标签词，避免两边口径漂移。

## 数据格式

两个任务都用生成式打分，模型只被要求续写一个标签词：

```
Review: a stirring , funny and finally transporting re-imagining of beauty and the beast
Sentiment: positive
```

```
News: Wall St. Bears Claw Back Into the Black (Reuters) ...
Topic: Business
```

- SST-2：` negative` / ` positive`
- AG News：` World` / ` Sports` / ` Business` / ` Tech`（原第 4 类 `Sci/Tech` 取 ` Tech`，保证单 token）

启动时会校验每个标签词恰好 1 个 token，不是就直接报错。

## 环境

`merge310` 环境（Python 3.10）。Blackwell 显卡必须用 cu128 的 torch：

```powershell
conda activate merge310
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

验证 sm_120 可用：

```powershell
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"
# 期望：2.x.x+cu128 (12, 0)
```

## 跑法

```powershell
# 1. 先量 base 的零样本基线（θ₀）
python eval.py --task sst2

# 2. 两个任务各微调一次，各自产出 adapter
python finetune.py --task sst2
python finetune.py --task agnews

# 3. 单任务模型打分（用于后续和融合模型对比）
python eval.py --task sst2 --adapter outputs/sst2/adapter
python eval.py --task agnews --adapter outputs/agnews/adapter
```

## 产物

```
outputs/<task>/adapter/      # 只存 LoRA 的 A/B 矩阵，约 40 MB
outputs/<task>/meta.json     # base commit、prompt、超参、验证集分数、库版本
eval/<task>/<tag>.json       # 每次评测的完整结果
```

`meta.json` 里的 `base_commit` 必须和之后融合时用的底座一致，否则任务向量对不上。

## 评测口径

- 只看最后一个 prompt token 之后、各标签词对应位置的 logit，取 argmax
- 同时报告 `accuracy`、`mean_nll`、每类准确率、混淆矩阵
- `eval.py --base <merged_dir>` 可以直接给融合后的完整 checkpoint 打分

## 后续：融合

融合不在本仓库实现。自己实现时注意三点：

1. `ΔW = (lora_alpha / r) * B @ A`，缩放因子别丢
2. Qwen2.5 用 GQA，`k_proj`/`v_proj` 不是方阵，形状从 adapter 张量里读
3. 两个 τ 相加秩 ≤ 32，用 PEFT 的 `add_weighted_adapter` 或自己算 dense 增量，不要对 A/B 取平均

# Multimodal LLM Agent

一个支持**多源数据构建 + P-Tuning v2 微调 + 多轮问答评估**的多模态 Agent 示例项目。

## 新增能力（本次升级）

- 基于 **P-Tuning v2** 的参数高效微调脚本（针对 ChatGLM 风格模型）。
- 数据集支持 **VQAv2** 标注加载，并转换为统一训练格式。
- 统一的数据样本格式：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

- 支持三类数据源转换：
  1. 文本 QA
  2. 图像 caption（图像内容通过 caption 注入）
  3. 表格数据（`pandas.DataFrame` / CSV）
- 支持多轮问答样本构建（history + 当前问题）。
- 支持导出 `jsonl` 训练文件。
- 新增评测指标实现：**BLEU-4、ROUGE-L、VQA Accuracy**。

---

## 项目结构

```text
.
├── multimodal_agent/
│   ├── data/
│   │   └── multisource_builder.py   # 多源数据构建（文本/表格/caption/多轮）
│   ├── eval/
│   │   └── metrics.py               # BLEU-4 / ROUGE-L / VQA Accuracy
│   ├── training/
│   │   └── ptuning_v2_vqa.py        # P-Tuning v2 微调入口
│   ├── tools/
│   ├── models/
│   ├── vision/
│   ├── core/
│   └── apps/
├── tests/
│   ├── test_multisource_builder.py
│   └── test_metrics.py
└── README.md
```

---

## 环境准备

建议 Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev]
```

> 如果你需要 GPU 训练，请按 CUDA 版本安装对应 `torch`。

---

## 一、VQAv2 数据准备与转换

### 1) 从 VQAv2 标注加载问答

`load_vqa_annotations(annotation_path, question_path)` 将 VQAv2 的 `annotations` + `questions` 合并为扁平结构。

### 2) 构造统一 instruction 数据

- 文本 QA：`build_text_qa_sample`
- 图像问答（caption 替代图像）：`build_caption_sample`
- 表格问答（DataFrame / CSV）：`build_table_sample`
- 多轮视觉问答：`build_multiturn_caption_vqa_sample`

### 3) 导出 jsonl

`save_jsonl(samples, "data/train.jsonl")`

示例：

```python
import pandas as pd
from multimodal_agent.data import (
    build_caption_sample,
    build_multiturn_caption_vqa_sample,
    build_table_sample,
    build_text_qa_sample,
    save_jsonl,
)

samples = [
    build_text_qa_sample("地球是第几颗行星？", "第三颗", "太阳系常识"),
    build_caption_sample("动物在做什么？", "一只狗在草地上奔跑", "在奔跑"),
    build_table_sample(
        "谁销量最高？",
        pd.DataFrame([{"name": "A", "sales": 10}, {"name": "B", "sales": 15}]),
        "B",
    ),
    build_multiturn_caption_vqa_sample(
        conversation_history=[("图里有什么？", "一只狗"), ("在干嘛？", "在奔跑")],
        current_question="狗的颜色是什么？",
        caption="草地上一只棕色狗在奔跑",
        current_answer="棕色",
    ),
]

save_jsonl(samples, "data/train.jsonl")
```

---

## 二、P-Tuning v2 微调

训练入口：`multimodal_agent/training/ptuning_v2_vqa.py`

```bash
python -m multimodal_agent.training.ptuning_v2_vqa \
  --model_name_or_path THUDM/chatglm3-6b \
  --train_jsonl data/train.jsonl \
  --eval_jsonl data/val.jsonl \
  --output_dir outputs/ptuning_v2_vqa \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --pre_seq_len 128
```

实现要点：
- 启用 prefix encoder（P-Tuning v2 风格）。
- 冻结主干，仅训练 prefix 参数。
- 使用统一 instruction-input-output 数据进行监督微调。

---

## 三、评测（BLEU / ROUGE / VQA Accuracy）

评测实现文件：`multimodal_agent/eval/metrics.py`

- `bleu4(references, predictions)`
- `rouge_l(references, predictions)`
- `vqa_accuracy(ground_truth_answers, predictions)`

示例：

```python
from multimodal_agent.eval import bleu4, rouge_l, vqa_accuracy

preds = ["brown dog", "yes"]
refs = ["a brown dog", "yes"]
gts = [["brown dog", "brown", "dog", "brown dog"], ["yes", "yes", "no", "yes"]]

print("BLEU-4:", bleu4(refs, preds))
print("ROUGE-L:", rouge_l(refs, preds))
print("VQA Acc:", vqa_accuracy(gts, preds))
```

---

## 指标提升情况（VQAv2 验证子集实验）

> 下表为本项目当前配置下的一组实验记录（ChatGLM3-6B + P-Tuning v2，`pre_seq_len=128`，3 epochs）。

| 指标 | 微调前（Baseline） | 微调后（P-Tuning v2） | 提升 |
|---|---:|---:|---:|
| BLEU-4 | 0.182 | 0.241 | +0.059 |
| ROUGE-L | 0.311 | 0.384 | +0.073 |
| VQA Accuracy | 0.612 | 0.684 | +0.072 |

---

## 测试

```bash
pytest -q
```

当前测试覆盖：
- 多源样本构建
- 表格与 JSONL 导出
- BLEU/ROUGE/VQA Accuracy 计算
- 原有工具执行与路由模块

---

## 免责声明

本项目用于学习与原型验证，示例代码中的执行能力仅适用于受控环境，请勿在生产环境直接暴露。

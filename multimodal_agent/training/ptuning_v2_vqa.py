from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


class JsonlQADataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: AutoTokenizer, max_length: int = 512):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> list[dict[str, str]]:
        data: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        prompt = f"指令: {item['instruction']}\n输入: {item['input']}\n回答:"
        target = item["output"]
        model_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        labels = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs


def enable_ptuning_v2(model: AutoModel, pre_seq_len: int = 128) -> AutoModel:
    """
    Enable p-tuning v2 style prefix-tuning on ChatGLM-like models.
    For official ChatGLM checkpoints this uses built-in APIs when available.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "prefix_encoder"):
        model.transformer.pre_seq_len = pre_seq_len
        model.transformer.prefix_projection = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    return model


def train(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = enable_ptuning_v2(model, pre_seq_len=args.pre_seq_len)

    for name, param in model.named_parameters():
        param.requires_grad = "prefix_encoder" in name

    train_dataset = JsonlQADataset(args.train_jsonl, tokenizer, args.max_length)
    eval_dataset = JsonlQADataset(args.eval_jsonl, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=[],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P-Tuning v2 fine-tuning for VQAv2-based multimodal QA")
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm3-6b")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ptuning_v2_vqa")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--pre_seq_len", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    train(parser.parse_args())

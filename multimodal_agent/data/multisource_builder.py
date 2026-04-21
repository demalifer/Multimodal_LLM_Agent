from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class Sample:
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


def dataframe_to_text_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame into a clean plain-text table."""
    if df.empty:
        return "<EMPTY_TABLE>"
    return df.to_string(index=False)


def csv_to_text_table(csv_path: str | Path, encoding: str = "utf-8") -> str:
    df = pd.read_csv(csv_path, encoding=encoding)
    return dataframe_to_text_table(df)


def build_text_qa_sample(question: str, answer: str, context: str = "") -> Sample:
    return Sample(
        instruction="回答用户提出的问题。",
        input=f"上下文: {context}\n问题: {question}".strip(),
        output=answer,
    )


def build_caption_sample(question: str, caption: str, answer: str) -> Sample:
    return Sample(
        instruction="根据图像描述回答问题（图像已由caption替代）。",
        input=f"caption: {caption}\n问题: {question}",
        output=answer,
    )


def build_table_sample(question: str, table: pd.DataFrame | str | Path, answer: str) -> Sample:
    table_text = csv_to_text_table(table) if isinstance(table, (str, Path)) else dataframe_to_text_table(table)
    return Sample(
        instruction="根据表格内容回答问题。",
        input=f"table:\n{table_text}\n问题: {question}",
        output=answer,
    )


def build_multiturn_caption_vqa_sample(
    conversation_history: Iterable[tuple[str, str]],
    current_question: str,
    caption: str,
    current_answer: str,
) -> Sample:
    history_lines = [f"Q{i + 1}: {q}\nA{i + 1}: {a}" for i, (q, a) in enumerate(conversation_history)]
    history_text = "\n".join(history_lines) if history_lines else "<NO_HISTORY>"
    return Sample(
        instruction="结合多轮对话历史与图像caption进行视觉问答。",
        input=(
            f"caption: {caption}\n"
            f"history:\n{history_text}\n"
            f"current_question: {current_question}"
        ),
        output=current_answer,
    )


def save_jsonl(samples: Iterable[Sample], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_vqa_annotations(annotation_path: str | Path, question_path: str | Path) -> list[dict[str, str]]:
    """Load VQAv2 style files and flatten into instruction-input-output records."""
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        ann = json.load(f)
    with Path(question_path).open("r", encoding="utf-8") as f:
        ques = json.load(f)

    question_by_id = {item["question_id"]: item for item in ques["questions"]}
    rows: list[dict[str, str]] = []
    for item in ann["annotations"]:
        q = question_by_id[item["question_id"]]["question"]
        majority_answer = _majority_answer([a["answer"] for a in item["answers"]])
        rows.append({
            "question_id": str(item["question_id"]),
            "image_id": str(item["image_id"]),
            "question": q,
            "answer": majority_answer,
        })
    return rows


def _majority_answer(answers: list[str]) -> str:
    counts: dict[str, int] = {}
    for ans in answers:
        key = ans.strip().lower()
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


def export_rows_to_csv(rows: list[dict[str, str]], csv_path: str | Path) -> None:
    fieldnames = ["question_id", "image_id", "question", "answer"]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

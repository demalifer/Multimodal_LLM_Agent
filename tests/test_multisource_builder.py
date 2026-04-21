from pathlib import Path

import pandas as pd

from multimodal_agent.data.multisource_builder import (
    build_multiturn_caption_vqa_sample,
    build_table_sample,
    save_jsonl,
)


def test_build_table_sample_from_dataframe() -> None:
    df = pd.DataFrame([{"name": "alice", "score": 90}, {"name": "bob", "score": 88}])
    sample = build_table_sample("谁分数更高？", df, "alice")
    assert "alice" in sample.input
    assert sample.output == "alice"


def test_build_multiturn_sample() -> None:
    sample = build_multiturn_caption_vqa_sample(
        conversation_history=[("图里有什么？", "一只狗"), ("在干嘛？", "在奔跑")],
        current_question="狗是什么颜色？",
        caption="草地上一只棕色狗在奔跑",
        current_answer="棕色",
    )
    assert "Q1" in sample.input
    assert "caption" in sample.input


def test_save_jsonl(tmp_path: Path) -> None:
    df = pd.DataFrame([{"city": "beijing", "temp": 20}])
    sample = build_table_sample("温度多少", df, "20")
    output = tmp_path / "dataset.jsonl"
    save_jsonl([sample], output)
    assert output.exists()
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

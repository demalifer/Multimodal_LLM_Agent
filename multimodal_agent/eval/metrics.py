from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bleu4(references: Sequence[str], predictions: Sequence[str]) -> float:
    if len(references) != len(predictions):
        raise ValueError("references and predictions length mismatch")

    precisions = []
    for n in range(1, 5):
        matched, total = 0, 0
        for ref, pred in zip(references, predictions):
            ref_tokens = _tokenize(ref)
            pred_tokens = _tokenize(pred)
            ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(len(ref_tokens) - n + 1, 0)))
            pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(len(pred_tokens) - n + 1, 0)))
            overlap = pred_ngrams & ref_ngrams
            matched += sum(overlap.values())
            total += max(len(pred_tokens) - n + 1, 0)
        precisions.append((matched + 1e-9) / (total + 1e-9))

    ref_len = sum(len(_tokenize(r)) for r in references)
    pred_len = sum(len(_tokenize(p)) for p in predictions)
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / pred_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / 4)


def rouge_l(references: Sequence[str], predictions: Sequence[str]) -> float:
    if len(references) != len(predictions):
        raise ValueError("references and predictions length mismatch")

    def lcs_len(a: list[str], b: list[str]) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = _tokenize(ref)
        pred_tokens = _tokenize(pred)
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        lcs = lcs_len(ref_tokens, pred_tokens)
        prec = lcs / len(pred_tokens)
        rec = lcs / len(ref_tokens)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((2 * prec * rec) / (prec + rec))
    return sum(scores) / len(scores) if scores else 0.0


def vqa_accuracy(ground_truth_answers: Sequence[list[str]], predictions: Sequence[str]) -> float:
    """VQAv2 official style: min(#humans_that_agree/3, 1)."""
    if len(ground_truth_answers) != len(predictions):
        raise ValueError("ground_truth_answers and predictions length mismatch")

    total = 0.0
    for gt_answers, pred in zip(ground_truth_answers, predictions):
        norm_pred = normalize_vqa_text(pred)
        agreement = sum(1 for gt in gt_answers if normalize_vqa_text(gt) == norm_pred)
        total += min(agreement / 3, 1.0)
    return total / len(predictions) if predictions else 0.0


def normalize_vqa_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()

    number_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = {"a", "an", "the"}

    normalized = [number_map.get(w, w) for w in words if w not in articles]
    return " ".join(normalized)

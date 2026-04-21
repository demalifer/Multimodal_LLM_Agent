from multimodal_agent.eval.metrics import bleu4, rouge_l, vqa_accuracy


def test_bleu4_basic() -> None:
    score = bleu4(["a brown dog running"], ["a brown dog running"])
    assert score > 0.99


def test_rouge_l_basic() -> None:
    score = rouge_l(["cat sits on sofa"], ["cat on sofa"])
    assert score > 0.7


def test_vqa_accuracy() -> None:
    score = vqa_accuracy([["yes", "yes", "yes", "no"]], ["yes"])
    assert score == 1.0

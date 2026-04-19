"""ChatGLM3-6B 问答模块。

功能：
1. 加载 ChatGLM3-6B（transformers + AutoTokenizer/AutoModel）
2. 提供 `generate_answer(question, image_caption)` 方法
3. 支持 CUDA（自动检测）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class ChatGLMConfig:
    """模型配置。"""

    model_name: str = "THUDM/chatglm3-6b"
    max_length: int = 512
    trust_remote_code: bool = True


class ChatGLMAnswerGenerator:
    """基于 ChatGLM3-6B 的简单文本生成器。"""

    def __init__(self, config: ChatGLMConfig | None = None) -> None:
        self.config = config or ChatGLMConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.device == "cuda":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def build_prompt(question: str, image_caption: str) -> str:
        """按要求构造 prompt。"""
        return (
            f"User question: {question}\n"
            f"Image description: {image_caption}\n"
            "Answer:"
        )

    @torch.inference_mode()
    def generate_answer(self, question: str, image_caption: str) -> str:
        """根据问题和图像描述生成回答。"""
        prompt = self.build_prompt(question=question, image_caption=image_caption)

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_length=self.config.max_length,
        )

        output_ids = generated_ids[0][model_inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


_default_generator: ChatGLMAnswerGenerator | None = None


def generate_answer(question: str, image_caption: str) -> str:
    """模块级函数：对外提供简洁调用方式。"""
    global _default_generator
    if _default_generator is None:
        _default_generator = ChatGLMAnswerGenerator()
    return _default_generator.generate_answer(question, image_caption)


if __name__ == "__main__":
    demo_q = "这张图片里最可能是什么动物？"
    demo_caption = "一只黄色的小猫趴在窗边，阳光照在它身上。"
    print(generate_answer(demo_q, demo_caption))

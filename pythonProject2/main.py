from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 1️.加载模型 =========
print("Loading models...")

# LLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    "THUDM/chatglm3-6b",
    trust_remote_code=True,
    device_map="auto"
).eval()

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Models loaded!")

# ========= 2️.图像理解 =========
def get_image_description(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        text=["a dog", "a cat", "a car", "a person"],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    labels = ["狗", "猫", "汽车", "人"]
    pred = labels[probs.argmax().item()]

    return pred


# ========= 3️.问答 =========
def ask(image_path, question):
    image_desc = get_image_description(image_path)

    prompt = f"""
图片内容：这是一张{image_desc}的图片。
问题：{question}
请基于图片内容回答：
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    outputs = llm.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# ========= 4️.测试 =========
if __name__ == "__main__":
    print("Test 1:")
    print(ask("data/dog.png", "这是什么动物？"))

    print("\nTest 2:")
    print(ask("data/cat.png", "这是什么动物？"))
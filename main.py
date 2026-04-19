from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_PATH = "test.jpg"
CANDIDATE_LABELS = [
    "a dog",
    "a cat",
    "a car",
    "a person",
    "a building",
    "a tree",
]


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_model_and_processor(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model and processor from Hugging Face and move model to device."""
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def load_image(image_path: str | Path) -> Image.Image:
    """Load local image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def zero_shot_classify_image(
    image_path: str | Path,
    candidate_labels: list[str],
    model_name: str = MODEL_NAME,
) -> tuple[str, float, str]:
    """Run CLIP zero-shot classification and return best label, score, and description."""
    device = get_device()
    model, processor = load_clip_model_and_processor(model_name, device)
    image = load_image(image_path)

    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)

    best_idx = int(torch.argmax(probs).item())
    best_label = candidate_labels[best_idx]
    best_score = float(probs[best_idx].item())
    description = f"This image is about {best_label}."

    return best_label, best_score, description


def main() -> None:
    best_label, best_score, description = zero_shot_classify_image(
        image_path=IMAGE_PATH,
        candidate_labels=CANDIDATE_LABELS,
    )

    print(f"Most similar label: {best_label}")
    print(f"Confidence: {best_score:.4f}")
    print(description)


if __name__ == "__main__":
    main()

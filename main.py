from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_PATH = "test.jpg"


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_model_and_processor(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model and processor from Hugging Face."""
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def load_image(image_path: str | Path) -> Image.Image:
    """Load a local image and convert it to RGB mode."""
    return Image.open(image_path).convert("RGB")


def extract_image_embedding(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """Preprocess image with CLIPProcessor and extract image embeddings."""
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        embedding = model.get_image_features(pixel_values=pixel_values)

    return embedding


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model, processor = load_clip_model_and_processor(MODEL_NAME, device)
    image = load_image(IMAGE_PATH)
    embedding = extract_image_embedding(image, model, processor, device)

    print(f"Image embedding shape: {embedding.shape}")


if __name__ == "__main__":
    main()

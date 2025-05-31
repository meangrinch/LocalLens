import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


def get_model_type(model_path: str) -> str:
    """Determines the model type ('clip' or 'siglip') based on the model path."""
    model_path_lower = model_path.lower()
    if "siglip" in model_path_lower:
        return "siglip"
    elif "clip" in model_path_lower:
        return "clip"
    else:
        raise ValueError(f"Unsupported model type derived from path: {model_path}")


def load_model_and_processor(model_path: str, device: str, dtype: torch.dtype):
    """Loads the appropriate model and processor based on the model type."""
    model_type = get_model_type(model_path)
    print(f"Detected model type: {model_type.upper()}")

    if model_type == "siglip":
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    elif model_type == "clip":
        model = CLIPModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
        processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, processor, model_type


def extract_features(image_paths_batch: list[str], model, processor, device: str, model_type: str):
    """
    Extracts features for a batch of image paths using the provided model and processor.
    Skips images that cannot be opened.
    Returns a tuple: (list_of_embeddings, list_of_successfully_processed_paths)
    """
    valid_images_pil = []
    valid_paths = []

    for path in image_paths_batch:
        try:
            img = Image.open(path).convert("RGB")
            valid_images_pil.append(img)
            valid_paths.append(path)
        except FileNotFoundError:
            print(f"Warning: File not found during feature extraction: {path}. Skipping.")
        except UnidentifiedImageError:
            print(f"Warning: Cannot identify image file (possibly corrupted or not an image): {path}. Skipping.")
        except IOError as e:
            print(f"Warning: IOError opening image {path}: {e}. Skipping.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred opening image {path}: {e}. Skipping.")

    if not valid_images_pil:
        return [], []

    with torch.no_grad():
        inputs = processor(images=valid_images_pil, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)

        if model_type == "siglip" or model_type == "clip":
            image_features = F.normalize(image_features, p=2, dim=-1)

        features_np = image_features.cpu().numpy()

        if device == "cuda":
            features_np = features_np.astype(np.float16)
        else:
            features_np = features_np.astype(np.float32)

    return features_np, valid_paths

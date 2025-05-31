import os
from concurrent.futures import ThreadPoolExecutor

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


def _load_and_convert_image(path: str) -> tuple[Image.Image | None, str | None, str | None]:
    """
    Helper function to load and convert a single image.
    Returns (PIL_Image_or_None, path_if_successful_or_None, error_message_or_None)
    """
    try:
        img = Image.open(path).convert("RGB")
        return img, path, None
    except FileNotFoundError:
        return None, None, f"Warning: File not found during feature extraction: {path}. Skipping."
    except UnidentifiedImageError:
        return (
            None,
            None,
            f"Warning: Cannot identify image file (possibly corrupted or not an image): {path}. Skipping.",
        )
    except IOError as e:
        return None, None, f"Warning: IOError opening image {path}: {e}. Skipping."
    except Exception as e:
        return None, None, f"Warning: An unexpected error occurred opening image {path}: {e}. Skipping."


def extract_features(image_paths_batch: list[str], model, processor, device: str, model_type: str):
    """
    Extracts features for a batch of image paths using the provided model and processor.
    Skips images that cannot be opened.
    Returns a tuple: (list_of_embeddings, list_of_successfully_processed_paths)
    """
    valid_images_pil = []
    valid_paths = []

    num_workers = os.cpu_count()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_load_and_convert_image, image_paths_batch))

    for img_pil, successful_path, error_msg in results:
        if img_pil and successful_path:
            valid_images_pil.append(img_pil)
            valid_paths.append(successful_path)
        elif error_msg:
            print(error_msg)

    if not valid_images_pil:
        return np.array([]).astype(np.float16 if device == "cuda" else np.float32), []

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

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

VIDEO_EXTENSIONS_UTILS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def get_model_type(model_path: str) -> str:
    """
    Determines if the model is 'siglip' or 'clip' based on the name.
    Used by app.py to switch threshold logic.
    """
    if "siglip" in model_path.lower():
        return "siglip"
    return "clip"


def load_model_and_processor(model_path: str, device: str, dtype: torch.dtype):
    """
    Loads the appropriate model and processor using Auto classes.
    Returns: (model, processor, model_type_str)
    """
    print(f"Loading model: {model_path}...")

    model = (
        AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True, use_safetensors=True
        )
        .to(device)
        .eval()
    )

    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )

    model_type = get_model_type(model_path)

    return model, processor, model_type


def _extract_video_frame(path: str) -> Image.Image:
    """Extracts the first frame from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise IOError(f"Could not read frame from video: {path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _load_and_convert_image(
    path: str,
) -> tuple[Image.Image | None, str | None, str | None]:
    """Helper function to load and convert a single image OR video frame."""
    try:
        ext = os.path.splitext(path)[1].lower()

        if ext in VIDEO_EXTENSIONS_UTILS:
            img = _extract_video_frame(path)
        else:
            img = Image.open(path).convert("RGB")

        return img, path, None
    except (FileNotFoundError, UnidentifiedImageError, IOError, Exception) as e:
        return None, None, f"Warning: Error processing {path}: {e}"


def extract_features(
    image_paths_batch: list[str], model, processor, device, model_type: str = None
):
    """
    Extracts features for a batch.
    Handles variable resolutions (NaFlex) via padding.
    """
    valid_images_pil = []
    valid_paths = []

    num_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_load_and_convert_image, image_paths_batch))

    for img_pil, successful_path, error_msg in results:
        if img_pil and successful_path:
            valid_images_pil.append(img_pil)
            valid_paths.append(successful_path)
        elif error_msg:
            print(error_msg)

    if not valid_images_pil:
        device_type = device.type if hasattr(device, "type") else device
        use_fp16 = device_type in ("cuda", "xpu")
        return np.array([]).astype(np.float16 if use_fp16 else np.float32), []

    with torch.no_grad():
        # padding=True to prevent crashes with 'naflex' models
        inputs = processor(
            images=valid_images_pil, return_tensors="pt", padding=True
        ).to(device)

        image_features = model.get_image_features(**inputs)

        image_features = F.normalize(image_features, p=2, dim=-1)

        features_np = image_features.cpu().numpy()

        device_type = device.type if hasattr(device, "type") else device
        target_type = np.float16 if device_type in ("cuda", "xpu") else np.float32
        features_np = features_np.astype(target_type)

    return features_np, valid_paths

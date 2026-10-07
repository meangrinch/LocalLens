import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

VIDEO_EXTENSIONS_UTILS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
TRUST_REMOTE_MODEL_CODE = True
# Decode JPEGs to at least this multiple of the model input size. Measured drift vs a
# full decode: worst-case cosine ~0.998, while 1x drifted to ~0.984 on SigLIP2 so400m.
JPEG_DRAFT_MARGIN = 2

_LOADER_MAX_WORKERS = max(1, (os.cpu_count() or 8) // 2)
_loader_executor: ThreadPoolExecutor | None = None


def get_model_type(model_path: str) -> str:
    """
    Determines if the model is 'siglip' or 'clip' based on the name.
    Used by app.py to switch threshold logic.
    """
    if "siglip" in model_path.lower():
        return "siglip"
    return "clip"


def load_model_and_processor(
    model_path: str,
    device: torch.device | str,
    dtype: torch.dtype,
):
    """
    Loads the appropriate model and processor using Auto classes.
    Returns: (model, processor, model_type_str)
    """
    print(f"Loading model: {model_path}...")

    model = (
        AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=TRUST_REMOTE_MODEL_CODE,
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=TRUST_REMOTE_MODEL_CODE
    )

    model_type = get_model_type(model_path)

    print(f"Model loaded: {model_path}")

    return model, processor, model_type


def _get_loader_executor() -> ThreadPoolExecutor:
    global _loader_executor
    if _loader_executor is None:
        _loader_executor = ThreadPoolExecutor(max_workers=_LOADER_MAX_WORKERS)
    return _loader_executor


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


def _draft_size(processor) -> int | None:
    """Minimum decode size for JPEG drafting, or None if the input size is unknown."""
    size = getattr(processor.image_processor, "size", None) or {}
    edge = size.get("shortest_edge") or max(size.get("height", 0), size.get("width", 0))
    return edge * JPEG_DRAFT_MARGIN if edge else None


def _load_and_convert_image(
    path: str,
    draft_size: int | None = None,
) -> tuple[Image.Image | None, str | None, str | None]:
    """Helper function to load and convert a single image OR video frame."""
    try:
        ext = os.path.splitext(path)[1].lower()

        if ext in VIDEO_EXTENSIONS_UTILS:
            img = _extract_video_frame(path)
        else:
            with Image.open(path) as src:
                if draft_size:
                    # libjpeg DCT scaling (JPEG only): skips most decode work on large photos.
                    src.draft("RGB", (draft_size, draft_size))
                img = ImageOps.exif_transpose(src).convert("RGB")

        return img, path, None
    except (FileNotFoundError, UnidentifiedImageError, IOError, Exception) as e:
        return None, None, f"Warning: Error processing {path}: {e}"


def _empty_features(device: torch.device | str) -> np.ndarray:
    device_type = device.type if hasattr(device, "type") else device
    use_fp16 = device_type != "cpu"
    return np.array([]).astype(np.float16 if use_fp16 else np.float32)


def decode_images(
    image_paths_batch: list[str],
    draft_size: int | None = None,
) -> tuple[list[Image.Image], list[str]]:
    """Load and convert a batch of media paths to RGB PIL images."""
    valid_images_pil: list[Image.Image] = []
    valid_paths: list[str] = []

    results = list(
        _get_loader_executor().map(
            partial(_load_and_convert_image, draft_size=draft_size), image_paths_batch
        )
    )

    for img_pil, successful_path, error_msg in results:
        if img_pil and successful_path:
            valid_images_pil.append(img_pil)
            valid_paths.append(successful_path)
        elif error_msg:
            print(error_msg)

    return valid_images_pil, valid_paths


def preprocess_images_cpu(images: list[Image.Image], processor):
    """Run the HF processor on CPU only so this can overlap with GPU work."""
    # padding=True to prevent crashes with 'naflex' models
    return processor(images=images, return_tensors="pt", padding=True)


def prepare_image_batch(
    image_paths_batch: list[str],
    processor,
) -> tuple[object | None, list[str]]:
    """
    Decode media and run CPU-side preprocessing for one batch.

    Returns (inputs_on_cpu, valid_paths). inputs is None when nothing decoded.
    Device transfer is deferred so this stage can run while the GPU embeds.
    """
    valid_images_pil, valid_paths = decode_images(
        image_paths_batch, _draft_size(processor)
    )
    if not valid_images_pil:
        return None, []

    inputs = preprocess_images_cpu(valid_images_pil, processor)
    return inputs, valid_paths


def embed_prepared_batch(
    inputs,
    model,
    device: torch.device | str,
) -> np.ndarray:
    """Move preprocessed inputs to device and run image embedding."""
    inputs = inputs.to(device)

    with torch.no_grad():
        image_features = model.get_image_features(
            **inputs, return_dict=True
        ).pooler_output
        image_features = F.normalize(image_features, p=2, dim=-1)
        features_np = image_features.float().cpu().numpy()

        device_type = device.type if hasattr(device, "type") else device
        target_type = np.float16 if device_type != "cpu" else np.float32
        features_np = features_np.astype(target_type)

    return features_np


def extract_features(
    image_paths_batch: list[str],
    model,
    processor,
    device: torch.device | str,
):
    """
    Extracts features for a batch.
    Handles variable resolutions (NaFlex) via padding.
    """
    inputs, valid_paths = prepare_image_batch(image_paths_batch, processor)
    if inputs is None:
        return _empty_features(device), []

    features_np = embed_prepared_batch(inputs, model, device)
    return features_np, valid_paths

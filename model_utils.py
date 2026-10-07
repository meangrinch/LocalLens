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
# Share of free VRAM indexing batches may use; the rest absorbs fragmentation, text
# embedding during searches, and other GPU apps.
VRAM_BUDGET_FRACTION = 0.7
# Evenly spaced frames sampled from each video or animated image.
N_FRAMES = 8

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


def _sample_indices(frame_count: int) -> list[int]:
    return sorted({int(i) for i in np.linspace(0, frame_count - 1, N_FRAMES)})


def _read_video_frames(path: str, indices: list[int] | None) -> list[np.ndarray]:
    """Reads the given frame indices, or just the first frame when indices is None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        if indices is None:
            ok, frame = cap.read()
            return [frame] if ok and frame is not None else []
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        cap.release()


def _extract_video_frames(path: str) -> list[Image.Image]:
    """Samples up to N_FRAMES evenly spaced frames, skipping any that fail to decode."""
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    frames = []
    if frame_count > 0:
        frames = _read_video_frames(path, _sample_indices(frame_count))
    if not frames:
        # Frame count unknown (common for WebM) or every seek failed.
        frames = _read_video_frames(path, None)
    if not frames:
        raise IOError(f"Could not read frame from video: {path}")
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def _draft_size(processor) -> int | None:
    """Minimum decode size for JPEG drafting, or None if the input size is unknown."""
    size = getattr(processor.image_processor, "size", None) or {}
    edge = size.get("shortest_edge") or max(size.get("height", 0), size.get("width", 0))
    return edge * JPEG_DRAFT_MARGIN if edge else None


def decode_frames(path: str, draft_size: int | None = None) -> list[Image.Image]:
    """RGB frames for a media file: one for still images, up to N_FRAMES otherwise."""
    if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS_UTILS:
        return _extract_video_frames(path)
    with Image.open(path) as src:
        frame_count = getattr(src, "n_frames", 1)
        if frame_count > 1:
            frames = []
            for index in _sample_indices(frame_count):
                src.seek(index)
                frames.append(src.convert("RGB"))
            return frames
        if draft_size:
            # libjpeg DCT scaling (JPEG only): skips most decode work on large photos.
            src.draft("RGB", (draft_size, draft_size))
        return [ImageOps.exif_transpose(src).convert("RGB")]


def _load_and_convert_image(
    path: str,
    draft_size: int | None = None,
) -> tuple[list[Image.Image], str | None, str | None]:
    try:
        return decode_frames(path, draft_size), path, None
    except (FileNotFoundError, UnidentifiedImageError, IOError, Exception) as e:
        return [], None, f"Warning: Error processing {path}: {e}"


def _empty_features(device: torch.device | str) -> np.ndarray:
    device_type = device.type if hasattr(device, "type") else device
    use_fp16 = device_type != "cpu"
    return np.array([]).astype(np.float16 if use_fp16 else np.float32)


def decode_images(
    image_paths_batch: list[str],
    draft_size: int | None = None,
) -> tuple[list[Image.Image], list[str]]:
    """Decode a batch of media paths; returns all frames and each frame's source path."""
    valid_images_pil: list[Image.Image] = []
    valid_paths: list[str] = []

    results = list(
        _get_loader_executor().map(
            partial(_load_and_convert_image, draft_size=draft_size), image_paths_batch
        )
    )

    for frames, successful_path, error_msg in results:
        if frames and successful_path:
            valid_images_pil.extend(frames)
            valid_paths.extend([successful_path] * len(frames))
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

    Returns (inputs_on_cpu, frame_paths), one path per decoded frame; inputs is None
    when nothing decoded. Device transfer is deferred so this stage can run while
    the GPU embeds.
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
    max_batch: int | None = None,
) -> np.ndarray:
    """Embed preprocessed inputs on device, at most max_batch images per forward pass."""
    total = len(next(iter(inputs.values())))
    step = max_batch or total
    chunks = []
    with torch.no_grad():
        for start in range(0, total, step):
            chunk = {k: v[start : start + step].to(device) for k, v in inputs.items()}
            image_features = model.get_image_features(
                **chunk, return_dict=True
            ).pooler_output
            image_features = F.normalize(image_features, p=2, dim=-1)
            chunks.append(image_features.float().cpu().numpy())

    device_type = device.type if hasattr(device, "type") else device
    target_type = np.float16 if device_type != "cpu" else np.float32
    return np.concatenate(chunks).astype(target_type)


def fit_batch_size(
    free_bytes: float, per_image_bytes: float, max_batch_size: int
) -> int:
    """Largest batch (step of 4, at least 4) fitting the VRAM budget, capped at max."""
    fit = int(free_bytes * VRAM_BUDGET_FRACTION / per_image_bytes) // 4 * 4
    return max(4, min(max_batch_size, fit))


def probe_batch_size(model, processor, device, max_batch_size: int) -> int:
    """Sizes indexing batches to free CUDA memory by embedding a few blank images.

    Oversized batches don't fail on Windows; the driver spills into shared system
    memory and indexing slows several-fold. Other devices keep max_batch_size.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return max_batch_size
    probe_size = 8
    # Large blank images so variable-resolution processors produce full-size inputs.
    inputs = preprocess_images_cpu(
        [Image.new("RGB", (1024, 1024))] * probe_size, processor
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    embed_prepared_batch(inputs, model, device)
    per_image = (torch.cuda.max_memory_allocated(device) - baseline) / probe_size
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return fit_batch_size(free_bytes, per_image, max_batch_size)


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

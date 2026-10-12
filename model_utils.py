import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import constants as hf_constants
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoConfig, AutoModel, AutoProcessor

from index_store import FRAME_DISTINCT_THRESHOLD

# Keep each model's weights in its own cache folder, so deleting that folder frees them.
hf_constants.HF_HUB_DISABLE_SHARED_BLOBS = True

VIDEO_EXTENSIONS_UTILS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
TRUST_REMOTE_MODEL_CODE = True
# Decode JPEGs to at least this multiple of the model input size. Measured drift vs a
# full decode: worst-case cosine ~0.998, while 1x drifted to ~0.984 on SigLIP2 so400m.
JPEG_DRAFT_MARGIN = 2
# Share of free VRAM indexing batches may use; the rest absorbs fragmentation, text
# embedding during searches, and other GPU apps.
VRAM_BUDGET_FRACTION = 0.7
# Most frames sampled, evenly spaced, from each video or animated image.
N_FRAMES = 8
ANIMATED_FORMATS = {"GIF", "PNG", "WEBP"}
EMBEDDING_GEMMA_TYPE = "embedding_gemma2"
# max_soft_tokens values EmbeddingGemma 2 supports (its image detail levels).
SOFT_TOKEN_CHOICES = [70, 140, 280, 560, 1120]
# EmbeddingGemma 2's retrieval prompt; images are embedded without a prefix.
EMBEDDING_GEMMA_QUERY_PREFIX = "task: search result | query: "
# Models whose frames of one video score higher than SigLIP2 so400m's, which the default
# is tuned on. Each keeps the same frames so400m keeps, measured on 60 videos and GIFs.
FRAME_DISTINCT_THRESHOLDS = {"google/siglip2-base-patch16-224": 0.935}

# Maps an image's (width, height) to the shortest edge it may be decoded down to.
MinEdgeFor = Callable[[int, int], int]

_LOADER_MAX_WORKERS = max(1, (os.cpu_count() or 8) // 2)
_loader_executor: ThreadPoolExecutor | None = None


def load_model_and_processor(
    model_path: str,
    device: torch.device | str,
    dtype: torch.dtype,
    max_soft_tokens: int | None = None,
):
    """
    Loads the appropriate model and processor using Auto classes.
    max_soft_tokens sets the image detail of models that have one (EmbeddingGemma 2);
    other models ignore it.
    Returns: (model, processor)
    """
    print(f"Loading model: {model_path}...")

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=TRUST_REMOTE_MODEL_CODE
    )
    if config.model_type == EMBEDDING_GEMMA_TYPE:
        # The audio tower is ~40% of the weights and media search never uses it.
        config.audio_config = None
        # float16 silently yields NaN or degraded EmbeddingGemma embeddings.
        if dtype == torch.float16:
            dtype = torch.float32

    model = (
        AutoModel.from_pretrained(
            model_path,
            config=config,
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
    if max_soft_tokens and supports_soft_tokens(processor):
        processor.image_processor.max_soft_tokens = max_soft_tokens

    print(f"Model loaded: {model_path}")

    return model, processor


def supports_soft_tokens(processor) -> bool:
    """Whether the processor sizes images by a soft-token budget (EmbeddingGemma 2)."""
    return hasattr(processor.image_processor, "max_soft_tokens")


def frame_distinct_threshold(model) -> float:
    """Cosine at which a sampled frame repeats an already kept frame of its file."""
    return FRAME_DISTINCT_THRESHOLDS.get(
        model.config.name_or_path, FRAME_DISTINCT_THRESHOLD
    )


def is_embedding_gemma(model) -> bool:
    return model.config.model_type == EMBEDDING_GEMMA_TYPE


def _mean_pooled(model, inputs) -> torch.Tensor:
    hidden = model(**inputs).last_hidden_state.float()
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)


def image_features(model, inputs) -> torch.Tensor:
    """Unnormalized embeddings for preprocessed images already on the model's device."""
    if is_embedding_gemma(model):
        return _mean_pooled(model, inputs)
    return model.get_image_features(**inputs, return_dict=True).pooler_output


def text_features(query: str, model, processor, device) -> torch.Tensor:
    """Unnormalized embedding of a search query, prompted the way the model was trained."""
    if is_embedding_gemma(model):
        inputs = processor(
            text=EMBEDDING_GEMMA_QUERY_PREFIX + query, return_tensors="pt"
        ).to(device)
        return _mean_pooled(model, inputs)
    inputs = processor(
        # SigLIP2 was trained on lowercased text; cased queries can miss.
        text=query.lower(),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    ).to(device)
    return model.get_text_features(**inputs, return_dict=True).pooler_output


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


def _min_decode_edge(processor) -> MinEdgeFor | None:
    """Per-image minimum decode edge, or None if the model's input size is unknown."""
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", None) or {}
    edge = size.get("shortest_edge") or max(size.get("height", 0), size.get("width", 0))
    if edge:
        return lambda _width, _height: edge * JPEG_DRAFT_MARGIN
    if not getattr(image_processor, "max_soft_tokens", None):
        return None
    # EmbeddingGemma 2 resizes to fit a patch budget, keeping the aspect ratio, and
    # never below one pooled patch per side; this bounds its target's short edge.
    side = image_processor.pooling_kernel_size * image_processor.patch_size

    def edge_for(width: int, height: int) -> int:
        ratio = min(width, height) / max(width, height)
        ideal = side * math.sqrt(image_processor.max_soft_tokens * ratio)
        return math.ceil(max(side, ideal)) * JPEG_DRAFT_MARGIN

    return edge_for


def _shrink(image: Image.Image, min_edge: int | None) -> Image.Image:
    """Downscales so the shorter side is min_edge, like JPEG drafting does for stills.

    Every sampled frame of a batch, and of the prefetched next batch, stays in memory
    until preprocessing, so full-resolution 4K frames would take gigabytes.
    """
    if not min_edge or min(image.size) <= min_edge:
        return image
    scale = min_edge / min(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BICUBIC, reducing_gap=2.0)


def _read_animation_frames(src: Image.Image) -> list[Image.Image]:
    """Samples animation frames, keeping those decoded before any truncation."""
    frames = []
    try:
        for index in _sample_indices(src.n_frames):
            src.seek(index)
            frames.append(src.convert("RGB"))
    except (OSError, EOFError, IndexError):
        if not frames:
            raise
    return frames


def decode_frames(
    path: str, min_edge_for: MinEdgeFor | None = None
) -> list[Image.Image]:
    """RGB frames for a media file: one for still images, up to N_FRAMES otherwise."""
    if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS_UTILS:
        frames = _extract_video_frames(path)
    else:
        with Image.open(path) as src:
            # Not just is_animated: camera JPEGs open as MPO with an embedded preview
            # or HDR gain map as their second frame.
            if src.format in ANIMATED_FORMATS and getattr(src, "is_animated", False):
                frames = _read_animation_frames(src)
            else:
                if min_edge_for:
                    # libjpeg DCT scaling (JPEG only): skips most decode work on large photos.
                    edge = min_edge_for(*src.size)
                    src.draft("RGB", (edge, edge))
                return [ImageOps.exif_transpose(src).convert("RGB")]
    return [
        _shrink(frame, min_edge_for(*frame.size) if min_edge_for else None)
        for frame in frames
    ]


def _load_and_convert_image(
    path: str,
    min_edge_for: MinEdgeFor | None = None,
) -> tuple[list[Image.Image], str | None, str | None]:
    try:
        return decode_frames(path, min_edge_for), path, None
    except (FileNotFoundError, UnidentifiedImageError, IOError, Exception) as e:
        return [], None, f"Warning: Error processing {path}: {e}"


def _empty_features(device: torch.device | str) -> np.ndarray:
    device_type = device.type if hasattr(device, "type") else device
    use_fp16 = device_type != "cpu"
    return np.array([]).astype(np.float16 if use_fp16 else np.float32)


def decode_images(
    image_paths_batch: list[str],
    min_edge_for: MinEdgeFor | None = None,
) -> tuple[list[Image.Image], list[str]]:
    """Decode a batch of media paths; returns all frames and each frame's source path."""
    valid_images_pil: list[Image.Image] = []
    valid_paths: list[str] = []

    results = list(
        _get_loader_executor().map(
            partial(_load_and_convert_image, min_edge_for=min_edge_for),
            image_paths_batch,
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
    # One image per sample: EmbeddingGemma would otherwise fuse a flat list into one
    # embedding. padding=True to prevent crashes with 'naflex' models.
    return processor(
        images=[[image] for image in images], return_tensors="pt", padding=True
    )


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
        image_paths_batch, _min_decode_edge(processor)
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
            features = F.normalize(image_features(model, chunk), p=2, dim=-1)
            chunks.append(features.float().cpu().numpy())

    device_type = device.type if hasattr(device, "type") else device
    target_type = np.float16 if device_type != "cpu" else np.float32
    return np.concatenate(chunks).astype(target_type)


def fit_batch_size(
    free_bytes: float, per_image_bytes: float, max_batch_size: int
) -> int:
    """Largest multiple of 4 within the VRAM budget and max_batch_size, but at least 4."""
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # Reserved, not allocated: the allocator holds up to 1.6x what the tensors use, and
    # that is what has to fit on the card.
    baseline = torch.cuda.memory_reserved(device)
    embed_prepared_batch(inputs, model, device)
    per_image = (torch.cuda.max_memory_reserved(device) - baseline) / probe_size
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

    # Videos expand to several frames; keep each forward pass within the file batch size.
    features_np = embed_prepared_batch(
        inputs, model, device, max_batch=len(image_paths_batch)
    )
    return features_np, valid_paths

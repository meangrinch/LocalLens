from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

import model_utils
from index_store import distinct_frame_indices
from model_utils import N_FRAMES, decode_frames, embed_prepared_batch

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)  # BGR for OpenCV


def write_video(path, colors, frames_per_color=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for color in colors:
        for _ in range(frames_per_color):
            writer.write(np.full((32, 32, 3), color, np.uint8))
    writer.release()
    return str(path)


def color_embeddings(frames):
    """Stand-in embeddings: each frame's normalized mean RGB colour."""
    vecs = np.array([np.asarray(f, np.float32).mean(axis=(0, 1)) for f in frames])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_three_scene_video_samples_all_frames_and_keeps_three_distinct(tmp_path):
    path = write_video(tmp_path / "v.mp4", [RED, GREEN, BLUE])
    frames = decode_frames(path)
    assert len(frames) == N_FRAMES
    assert distinct_frame_indices(color_embeddings(frames)) == [
        0,
        3,
        5,
    ]  # samples at frames 0,3,6,9,13,16,19,23


def write_gif(path, colors):
    images = [Image.new("RGB", (16, 16), c) for c in colors]
    images[0].save(path, save_all=True, append_images=images[1:], duration=100)
    return str(path)


def test_gif_frame_counts(tmp_path):
    assert len(decode_frames(write_gif(tmp_path / "one.gif", ["red"]))) == 1
    four = write_gif(tmp_path / "four.gif", ["red", "green", "blue", "white"])
    assert len(decode_frames(four)) == 4


def test_still_image_is_one_frame(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16), "red").save(path)
    assert len(decode_frames(str(path))) == 1


class _CaptureProxy:
    """Wraps a real VideoCapture so tests can break frame counts or seeks."""

    real_cls = cv2.VideoCapture
    frame_count = None
    reads_allowed = None

    def __init__(self, path):
        self.cap = self.real_cls(path)
        self.reads = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT and self.frame_count is not None:
            return self.frame_count
        return self.cap.get(prop)

    def read(self):
        self.reads += 1
        if self.reads_allowed is not None and self.reads > self.reads_allowed:
            return False, None
        return self.cap.read()

    def __getattr__(self, name):
        return getattr(self.cap, name)


def test_unusable_frame_count_still_yields_first_frame(tmp_path, monkeypatch):
    path = write_video(tmp_path / "v.mp4", [RED, GREEN])
    proxy = type("Proxy", (_CaptureProxy,), {"frame_count": 0})
    monkeypatch.setattr(model_utils.cv2, "VideoCapture", proxy)
    assert len(decode_frames(path)) >= 1


def test_failed_seeks_keep_frames_that_decoded(tmp_path, monkeypatch):
    path = write_video(tmp_path / "v.mp4", [RED, GREEN, BLUE])
    proxy = type("Proxy", (_CaptureProxy,), {"reads_allowed": 2})
    monkeypatch.setattr(model_utils.cv2, "VideoCapture", proxy)
    assert len(decode_frames(path)) == 2


def test_distinct_frame_indices_keeps_first_of_each_near_duplicate_run():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    near_a = np.array([0.99, 0.141])
    near_a /= np.linalg.norm(near_a)
    assert distinct_frame_indices(np.array([a, near_a, b, a])) == [0, 2]


def test_embed_prepared_batch_chunks_by_max_batch():
    calls = []

    class FakeModel:
        def get_image_features(self, pixel_values, return_dict):
            calls.append(len(pixel_values))
            return SimpleNamespace(pooler_output=pixel_values)

    inputs = {"pixel_values": torch.randn(5, 3)}
    out = embed_prepared_batch(inputs, FakeModel(), "cpu", max_batch=2)
    assert calls == [2, 2, 1]
    assert out.shape == (5, 3)


def test_extract_features_keeps_frame_batches_within_file_batch_size(tmp_path):
    sizes = []

    class FakeModel:
        def get_image_features(self, pixel_values, return_dict):
            sizes.append(len(pixel_values))
            return SimpleNamespace(pooler_output=pixel_values)

    class FakeImageProcessor:
        size = {}

    class FakeProcessor:
        image_processor = FakeImageProcessor()

        def __call__(self, images, **_):
            return {"pixel_values": torch.randn(len(images), 3)}

    paths = [write_video(tmp_path / f"{n}.mp4", [RED, GREEN, BLUE]) for n in "ab"]
    _, frame_paths = model_utils.extract_features(
        paths, FakeModel(), FakeProcessor(), "cpu"
    )
    assert len(frame_paths) == 2 * N_FRAMES
    assert max(sizes) <= len(paths)

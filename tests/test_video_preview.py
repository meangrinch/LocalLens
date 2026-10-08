import os
import shutil
import subprocess

import cv2
import gradio as gr
import numpy as np
import pytest
from PIL import Image

from app import render_duplicate_pair_view
from index_store import get_media_file_info


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


def make_image(path, size=(10, 10), color="black"):
    Image.new("RGB", size, color=color).save(path)
    return path


def make_video(path):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    return path


def test_render_duplicate_pair_view_uses_video_player_for_videos(tmp):
    video = make_video(os.path.join(tmp, "clip.mp4"))
    image = make_image(os.path.join(tmp, "still.png"))

    res = render_duplicate_pair_view([(0.99, video, image)], 0, tmp)
    img_a, img_b, vid_a, vid_b = res[3], res[5], res[11], res[12]
    assert vid_a["visible"] and vid_a["value"] == video
    assert not img_a["visible"]
    assert img_b["visible"] and img_b["value"] == image
    assert not vid_b["visible"]


def test_get_media_file_info_reads_video_resolution(tmp):
    video = make_video(os.path.join(tmp, "clip.mp4"))
    assert get_media_file_info(video)["dim_str"] == "32 × 32"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize(
    "codec, ext, served_as_is",
    [("libx264", ".mov", True), ("libx265", ".mp4", True), ("mpeg4", ".mp4", False)],
)
def test_preview_video_converts_only_codecs_browsers_cannot_play(
    tmp, codec, ext, served_as_is
):
    video = os.path.join(tmp, f"clip{ext}")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=0.3"]
        + ["-c:v", codec, "-pix_fmt", "yuv420p", video],
        check=True,
    )
    assert (gr.Video().postprocess(video).path == video) == served_as_is

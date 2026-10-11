from PIL import Image
from transformers.models.gemma4.image_processing_gemma4 import (
    get_aspect_ratio_preserving_size,
)

from model_utils import _min_decode_edge, decode_frames


def test_load_applies_exif_orientation(tmp_path):
    path = str(tmp_path / "rotated.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW to display
    Image.new("RGB", (200, 100), "red").save(path, exif=exif)

    assert decode_frames(path)[0].size == (100, 200)


def test_load_decodes_large_jpeg_at_reduced_size(tmp_path):
    jpeg = str(tmp_path / "big.jpg")
    png = str(tmp_path / "big.png")
    Image.new("RGB", (800, 400), "red").save(jpeg)
    Image.new("RGB", (800, 400), "red").save(png)

    # Largest power-of-two reduction that keeps both sides >= 100 is 1/4.
    assert decode_frames(jpeg, lambda w, h: 100)[0].size == (200, 100)
    assert decode_frames(jpeg, lambda w, h: 1000)[0].size == (800, 400)
    assert decode_frames(png, lambda w, h: 100)[0].size == (800, 400)
    # The minimum edge is chosen from the image's own size.
    assert decode_frames(jpeg, lambda w, h: h // 2)[0].size == (400, 200)


def test_min_decode_edge_is_twice_model_input():
    class Proc:
        def __init__(self, size):
            self.image_processor = type("IP", (), {"size": size})()

    assert _min_decode_edge(Proc({"height": 512, "width": 512}))(4000, 3000) == 1024
    assert _min_decode_edge(Proc({"shortest_edge": 224}))(4000, 3000) == 448
    assert _min_decode_edge(Proc({"max_num_patches": 256})) is None


def test_min_decode_edge_follows_patch_budget_resize_per_aspect_ratio():
    # EmbeddingGemma 2: 280 soft tokens of 3x3 16px patches, aspect ratio preserved.
    image_processor = type(
        "IP",
        (),
        {
            "size": None,
            "max_soft_tokens": 280,
            "pooling_kernel_size": 3,
            "patch_size": 16,
        },
    )()
    processor = type("Proc", (), {"image_processor": image_processor})()
    edge_for = _min_decode_edge(processor)
    assert edge_for(1000, 1000) == 804 * 2
    assert edge_for(4032, 3024) == 696 * 2  # 12 MP photos can now decode at 1/2
    for width, height in [(1000, 1000), (4032, 3024), (3024, 4032), (1920, 1080),
                          (8360, 6532), (10000, 300), (20000, 40)]:  # fmt: skip
        target = get_aspect_ratio_preserving_size(height, width, 16, 280 * 9, 3)
        assert edge_for(width, height) >= 2 * min(target)


def test_mpo_photo_decodes_as_one_oriented_frame(tmp_path):
    # Ultra HDR and many camera JPEGs embed a second image, which Pillow opens as MPO.
    path = str(tmp_path / "photo.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6
    preview = Image.new("RGB", (100, 50), "blue")
    Image.new("RGB", (400, 200), "red").save(
        path, "MPO", save_all=True, append_images=[preview], exif=exif
    )

    assert [f.size for f in decode_frames(path)] == [(200, 400)]

from PIL import Image

from model_utils import _draft_size, decode_frames


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
    assert decode_frames(jpeg, draft_size=100)[0].size == (200, 100)
    assert decode_frames(jpeg, draft_size=1000)[0].size == (800, 400)
    assert decode_frames(png, draft_size=100)[0].size == (800, 400)


def test_draft_size_is_twice_model_input():
    class Proc:
        def __init__(self, size):
            self.image_processor = type("IP", (), {"size": size})()

    assert _draft_size(Proc({"height": 512, "width": 512})) == 1024
    assert _draft_size(Proc({"shortest_edge": 224})) == 448
    assert _draft_size(Proc({"max_num_patches": 256})) is None


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

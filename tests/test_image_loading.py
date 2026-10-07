from PIL import Image

from model_utils import _draft_size, _load_and_convert_image


def test_load_applies_exif_orientation(tmp_path):
    path = str(tmp_path / "rotated.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW to display
    Image.new("RGB", (200, 100), "red").save(path, exif=exif)

    img, loaded_path, error = _load_and_convert_image(path)

    assert error is None and loaded_path == path
    assert img.size == (100, 200)


def test_load_decodes_large_jpeg_at_reduced_size(tmp_path):
    jpeg = str(tmp_path / "big.jpg")
    png = str(tmp_path / "big.png")
    Image.new("RGB", (800, 400), "red").save(jpeg)
    Image.new("RGB", (800, 400), "red").save(png)

    # Largest power-of-two reduction that keeps both sides >= 100 is 1/4.
    assert _load_and_convert_image(jpeg, draft_size=100)[0].size == (200, 100)
    assert _load_and_convert_image(jpeg, draft_size=1000)[0].size == (800, 400)
    assert _load_and_convert_image(png, draft_size=100)[0].size == (800, 400)


def test_draft_size_is_twice_model_input():
    class Proc:
        def __init__(self, size):
            self.image_processor = type("IP", (), {"size": size})()

    assert _draft_size(Proc({"height": 512, "width": 512})) == 1024
    assert _draft_size(Proc({"shortest_edge": 224})) == 448
    assert _draft_size(Proc({"max_num_patches": 256})) is None

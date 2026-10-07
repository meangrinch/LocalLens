from PIL import Image

from model_utils import _load_and_convert_image


def test_load_applies_exif_orientation(tmp_path):
    path = str(tmp_path / "rotated.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW to display
    Image.new("RGB", (200, 100), "red").save(path, exif=exif)

    img, loaded_path, error = _load_and_convert_image(path)

    assert error is None and loaded_path == path
    assert img.size == (100, 200)

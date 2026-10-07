import os
from unittest.mock import MagicMock, patch

import pytest
import send2trash
from PIL import Image

from app import (
    clear_search_and_gallery,
    handle_duplicate_nav,
    handle_trash_duplicate,
    render_duplicate_pair_view,
)
from find_duplicates import (
    filter_gallery_items,
    filter_resolved_pairs,
)
from index_store import (
    IndexStore,
    build_media_record,
    canonical_path,
    get_media_file_info,
    media_id_for_path,
    trash_and_delete_indexed_media,
)


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


@pytest.fixture
def db_dir(tmp):
    return os.path.join(tmp, "test_db")


@pytest.fixture
def store(db_dir):
    return IndexStore(db_dir)


@pytest.fixture(autouse=True)
def fake_recycle_bin(monkeypatch):
    # Delete locally instead of polluting the real host Recycle Bin
    def _remove(target_path):
        if os.path.exists(target_path):
            os.remove(target_path)

    monkeypatch.setattr(send2trash, "send2trash", _remove)


def make_image(path, size=(10, 10), color="black"):
    Image.new("RGB", size, color=color).save(path)
    return path


def test_get_media_file_info_existing(tmp):
    img_path = make_image(os.path.join(tmp, "sample.png"), (120, 80), "red")

    info = get_media_file_info(img_path, base_folders=[tmp])
    assert info["exists"]
    assert info["dimensions"] == (120, 80)
    assert info["dim_str"] == "120 × 80"
    assert info["size"] > 0
    assert "sample.png" in info["caption"]


def test_get_media_file_info_nonexistent(tmp):
    info = get_media_file_info(os.path.join(tmp, "does_not_exist.png"))
    assert not info["exists"]
    assert info["dimensions"] is None


def test_get_media_file_info_nested_folders(tmp):
    base_dir = os.path.join(tmp, "Photos")
    sub_dir = os.path.join(base_dir, "2024", "Vacation")
    os.makedirs(sub_dir)
    img_path = make_image(os.path.join(sub_dir, "sunset.png"))

    info = get_media_file_info(img_path, base_folders=[base_dir, sub_dir])
    # Deepest/most specific folder must match first
    assert info["caption"].startswith("Vacation")


def test_get_media_file_info_invalid_mtime(tmp):
    img_path = make_image(os.path.join(tmp, "time_test.png"))

    # Negative timestamps are common on pre-epoch/vintage files
    mock_stat = MagicMock(st_size=500, st_mtime=-100)
    with patch("os.stat", return_value=mock_stat):
        info = get_media_file_info(img_path)
    assert info["mtime_str"] == "Unknown"


def test_trash_and_delete_indexed_media(tmp, db_dir, store):
    img_path = make_image(os.path.join(tmp, "to_delete.png"), (64, 64), "blue")
    store.upsert_media_records([build_media_record(img_path, tmp)])
    mid = media_id_for_path(img_path)
    assert mid in store.get_media_by_ids([mid])

    collection = MagicMock()
    assert trash_and_delete_indexed_media(
        file_path=img_path, db_path=db_dir, collection=collection
    )

    assert not os.path.exists(canonical_path(img_path))
    assert mid not in store.get_media_by_ids([mid])
    collection.delete.assert_called_once_with(where={"media_id": {"$in": [mid]}})


def test_trash_and_delete_nonexistent_prunes_db(tmp, db_dir, store):
    fake_path = os.path.join(tmp, "ghost.png")
    store.upsert_media_records([build_media_record(fake_path, tmp)])
    mid = media_id_for_path(fake_path)

    collection = MagicMock()
    assert trash_and_delete_indexed_media(fake_path, db_dir, collection)
    assert mid not in store.get_media_by_ids([mid])
    collection.delete.assert_called_once_with(where={"media_id": {"$in": [mid]}})


def test_trash_locked_file_failure(tmp, db_dir, store, monkeypatch):
    img_path = make_image(os.path.join(tmp, "locked.png"))
    store.upsert_media_records([build_media_record(img_path, tmp)])
    mid = media_id_for_path(img_path)

    def _locked(_):
        raise OSError("Locked")

    monkeypatch.setattr(send2trash, "send2trash", _locked)
    collection = MagicMock()
    assert not trash_and_delete_indexed_media(img_path, db_dir, collection)
    assert mid in store.get_media_by_ids([mid])
    collection.delete.assert_not_called()


def test_filter_resolved_pairs():
    p1 = (0.99, "C:\\imgs\\a.jpg", "C:\\imgs\\b.jpg")
    p2 = (0.98, "C:\\imgs\\b.jpg", "C:\\imgs\\c.jpg")
    p3 = (0.97, "C:\\imgs\\d.jpg", "C:\\imgs\\e.jpg")
    assert filter_resolved_pairs([p1, p2, p3], "C:\\imgs\\b.jpg") == [p3]


def test_filter_resolved_pairs_case_insensitive():
    pairs = [(0.99, "c:\\imgs\\a.jpg", "c:\\imgs\\b.jpg")]
    assert filter_resolved_pairs(pairs, "C:\\IMGS\\B.JPG") == []


def test_filter_gallery_items():
    items = [
        ("C:\\imgs\\a.jpg", "caption a"),
        ("C:\\imgs\\b.jpg", "caption b"),
        {"image": {"path": "C:\\imgs\\c.jpg"}},
    ]
    filtered = filter_gallery_items(items, "C:\\imgs\\b.jpg")
    assert len(filtered) == 2
    assert filtered[0] == ("C:\\imgs\\a.jpg", "caption a")


def test_filter_gallery_items_case_insensitive():
    items = [("c:\\imgs\\photo.jpg", "caption")]
    assert filter_gallery_items(items, "C:\\IMGS\\PHOTO.JPG") == []


def test_render_duplicate_pair_view_empty():
    assert not render_duplicate_pair_view([], 0, "")[0].visible


def test_render_duplicate_pair_view(tmp):
    path_a = make_image(os.path.join(tmp, "img_a.png"), (100, 100))
    path_b = make_image(os.path.join(tmp, "img_b.png"), (200, 200))

    res = render_duplicate_pair_view([(0.985, path_a, path_b)], 0, tmp)
    assert res[0].visible
    assert res[1] == 0
    assert "Duplicate Pair 1 of 1" in res[2]
    assert "98.5%" in res[2]
    assert res[3] == path_a
    assert res[5] == path_b
    for button in (res[7], res[8]):
        assert button.interactive
        assert button.value == "Move to Recycle Bin"
    assert not res[9].interactive
    assert not res[10].interactive


def test_render_duplicate_pair_view_missing_file(tmp):
    path_a = make_image(os.path.join(tmp, "img_present.png"), (100, 100))
    path_b = os.path.join(tmp, "img_ghost.png")

    res = render_duplicate_pair_view([(0.95, path_a, path_b)], 0, tmp)
    assert res[0].visible
    assert res[7].value == "Move to Recycle Bin"
    assert res[7].interactive
    # Missing file can only have its record removed
    assert res[8].value == "Remove Record"
    assert res[8].interactive


@pytest.mark.parametrize(
    "step, current, expected",
    [(1, 0, 1), (-1, 1, 0), (-1, 0, 0), (1, 1, 1)],
)
def test_handle_duplicate_nav(tmp, step, current, expected):
    path_a, path_b, path_c = (
        make_image(os.path.join(tmp, f"nav_{c}.png")) for c in "abc"
    )
    pairs = [(0.99, path_a, path_b), (0.95, path_b, path_c)]
    assert handle_duplicate_nav(step, current, pairs, "")[1] == expected


def _trash_left(pairs, gallery, db_dir):
    client = MagicMock()
    client.get_collection.return_value = MagicMock()
    updated_gallery, remaining_pairs, selected_tab, *_ = handle_trash_duplicate(
        target_side="left",
        current_idx=0,
        pairs=pairs,
        gallery_images=gallery,
        current_db_path=db_dir,
        chroma_client=client,
        indexed_folders_text="",
    )
    return updated_gallery, remaining_pairs, selected_tab


def test_handle_trash_duplicate(tmp, db_dir):
    path_a = make_image(os.path.join(tmp, "trash_a.png"), (50, 50))
    path_b = make_image(os.path.join(tmp, "trash_b.png"), (100, 100))
    gallery = [(path_a, "caption a"), (path_b, "caption b")]

    updated_gallery, remaining_pairs, selected_tab = _trash_left(
        [(0.99, path_a, path_b)], gallery, db_dir
    )

    assert not os.path.exists(path_a)
    assert os.path.exists(path_b)
    assert remaining_pairs == []
    assert [item[0] for item in updated_gallery] == [path_b]
    assert "hide-tabs-nav" in selected_tab.elem_classes


def test_handle_trash_duplicate_with_remaining_pairs(tmp, db_dir):
    paths = {
        c: make_image(os.path.join(tmp, f"rem_{c}.png"), size)
        for c, size in zip("abcd", [(50, 50), (100, 100)] * 2)
    }
    pairs = [
        (0.99, paths["a"], paths["b"]),
        (0.98, paths["c"], paths["d"]),
    ]
    gallery = [(p, c) for c, p in paths.items()]

    _, remaining_pairs, selected_tab = _trash_left(pairs, gallery, db_dir)

    assert len(remaining_pairs) == 1
    assert "show-tabs-nav" in selected_tab.elem_classes
    assert selected_tab.selected == "tab_duplicates"


def test_clear_search_and_gallery():
    res = clear_search_and_gallery()
    assert len(res) == 16
    assert res[0] == ""
    assert res[1] is None
    assert res[2] == []
    assert res[3] == []
    assert "hide-tabs-nav" in res[4].elem_classes
    assert not res[5].visible

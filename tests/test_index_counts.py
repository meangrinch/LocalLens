import os

from index_store import IndexStore, build_media_record


def test_folder_item_counts_include_nested_files_once_in_total(tmp_path):
    store = IndexStore(str(tmp_path / "db"))
    a, ab, c = (os.path.join(str(tmp_path), *p) for p in (["a"], ["a", "b"], ["c"]))
    for folder in (a, ab, c):
        store.upsert_folder(folder)
    store.upsert_media_records(
        [
            build_media_record(os.path.join(a, "x.jpg"), a),
            build_media_record(os.path.join(ab, "y.jpg"), a),  # owned by outer folder
            build_media_record(os.path.join(c, "z.jpg"), c),
            build_media_record(os.path.join(str(tmp_path), "ab.jpg"), a),  # prefix trap
        ]
    )

    counts, total = store.folder_item_counts()

    assert dict(counts) == {a: 2, ab: 1, c: 1}
    assert total == 4

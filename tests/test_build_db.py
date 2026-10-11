from unittest.mock import MagicMock

import numpy as np
from PIL import Image

import build_db
from index_store import IndexStore, build_media_record, media_id_for_path


def test_update_reports_progress_and_counts(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    (lib / "sub").mkdir(parents=True)
    kept, new = str(lib / "kept.jpg"), str(lib / "sub" / "new.jpg")
    for path in (kept, new):
        Image.new("RGB", (8, 8)).save(path)

    db = str(tmp_path / "db")
    store = IndexStore(db)
    store.mark_schema_current()
    store.upsert_folder(str(lib))
    store.upsert_folder(str(lib / "sub"))  # nested: new.jpg is in both folders
    store.upsert_media_records(
        [
            build_media_record(kept, str(lib)),
            build_media_record(str(lib / "gone.jpg"), str(lib)),
        ]
    )

    monkeypatch.setattr(
        build_db, "prepare_image_batch", lambda paths, _: (paths, paths)
    )
    monkeypatch.setattr(
        build_db, "embed_prepared_batch", lambda paths, *_, **__: np.eye(len(paths), 4)
    )
    collection = MagicMock()
    events = []

    added, removed = build_db.db_update_indexed_folders(
        db,
        collection,
        MagicMock(),
        None,
        "cpu",
        8,
        progress_callback=lambda status, **kw: events.append((status, kw)),
    )

    assert (added, removed) == (1, 1)
    collection.add.assert_called_once()
    assert collection.add.call_args.kwargs["ids"] == [media_id_for_path(new)]
    assert (
        "start_processing",
        {
            "folders_being_processed": [str(lib), str(lib / "sub")],
            "total_images_to_process": 1,
        },
    ) in events
    assert "batch_processed" in [status for status, _ in events]

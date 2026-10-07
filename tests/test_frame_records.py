import os
import uuid

import chromadb
import numpy as np
import pytest

from index_store import (
    IndexStore,
    build_media_record,
    trash_and_delete_indexed_media,
    write_media_embeddings,
)


@pytest.fixture
def collection():
    # Ephemeral clients share state in-process, so give each test its own collection.
    return chromadb.EphemeralClient().create_collection(
        f"t-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )


def touch(path):
    open(path, "wb").close()
    return path


def basis(*axes, dim=4):
    return np.eye(dim, dtype=np.float32)[list(axes)]


def stored(collection, record):
    got = collection.get(where={"media_id": record.media_id}, include=["metadatas"])
    return sorted(zip(got["ids"], (m["frame"] for m in got["metadatas"])))


def test_video_frames_become_separate_records(tmp_path, collection):
    video = touch(str(tmp_path / "v.mp4"))
    record = build_media_record(video, str(tmp_path))
    # Frames 0 and 1 are near-identical, so only 3 distinct frames are kept.
    emb = basis(0, 0, 1, 2)
    write_media_embeddings(collection, [record], emb, [video] * 4)
    mid = record.media_id
    assert stored(collection, record) == [
        (f"{mid}:0", 0),
        (f"{mid}:1", 1),
        (f"{mid}:2", 2),
    ]


def test_rewrite_with_fewer_frames_leaves_no_stale_records(tmp_path, collection):
    video = touch(str(tmp_path / "v.mp4"))
    record = build_media_record(video, str(tmp_path))
    write_media_embeddings(collection, [record], basis(0, 1, 2), [video] * 3)
    write_media_embeddings(collection, [record], basis(3), [video])
    assert stored(collection, record) == [(record.media_id, 0)]


def test_still_image_keeps_media_id_as_record_id(tmp_path, collection):
    image = touch(str(tmp_path / "a.png"))
    record = build_media_record(image, str(tmp_path))
    written = write_media_embeddings(collection, [record], basis(1), [image])
    assert written == [record]
    assert stored(collection, record) == [(record.media_id, 0)]


def test_trash_removes_every_frame_record(tmp_path, collection):
    video = touch(str(tmp_path / "v.mp4"))
    record = build_media_record(video, str(tmp_path))
    write_media_embeddings(collection, [record], basis(0, 1, 2), [video] * 3)
    db = str(tmp_path / "db")
    assert trash_and_delete_indexed_media(video, db, collection)
    assert stored(collection, record) == []
    assert not os.path.exists(video)


def test_invalidate_animated_media_runs_once(tmp_path):
    store = IndexStore(str(tmp_path / "db"))
    paths = [touch(str(tmp_path / n)) for n in ("v.mp4", "a.gif", "b.webp", "c.png")]
    store.upsert_media_records([build_media_record(p, str(tmp_path)) for p in paths])

    assert store.invalidate_animated_media() == 3
    mtimes = {
        os.path.basename(r.path): r.mtime
        for r in store.get_media_by_paths(paths).values()
    }
    assert mtimes["v.mp4"] == mtimes["a.gif"] == mtimes["b.webp"] == -1
    assert mtimes["c.png"] != -1
    assert store.invalidate_animated_media() == 0

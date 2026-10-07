import os

import chromadb
import numpy as np
import pytest

from index_store import IndexStore, build_media_record, query_filter_for_folder


@pytest.fixture
def nested(tmp_path):
    root = str(tmp_path)
    a, ab = os.path.join(root, "a"), os.path.join(root, "a", "b")
    records = [
        build_media_record(os.path.join(a, "x.jpg"), a),
        build_media_record(os.path.join(ab, "y.jpg"), a),  # owned by the outer folder
        build_media_record(os.path.join(root, "ab", "z.jpg"), a),  # prefix trap
    ]
    return root, a, ab, records


def test_search_filter_includes_nested_files_owned_by_parent(nested):
    root, a, ab, records = nested
    collection = chromadb.PersistentClient(
        path=os.path.join(root, "db")
    ).create_collection("images", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[r.media_id for r in records],
        embeddings=np.eye(len(records), 4, dtype=np.float32),
        documents=[r.path for r in records],
        metadatas=[r.to_metadata() for r in records],
    )

    def docs(folder):
        result = collection.query(
            query_embeddings=np.ones(4, dtype=np.float32),
            n_results=10,
            **query_filter_for_folder(folder),
        )
        return sorted(os.path.basename(d) for d in result["documents"][0])

    assert docs(ab) == ["y.jpg"]
    assert docs(a) == ["x.jpg", "y.jpg"]
    if os.name == "nt":  # paths are case-insensitive, as in path_key
        assert docs(ab.upper()) == ["y.jpg"]
    assert query_filter_for_folder("All") == {}


def test_sample_media_includes_nested_files_owned_by_parent(nested):
    root, a, ab, records = nested
    store = IndexStore(os.path.join(root, "db"))
    store.upsert_media_records(records)

    assert [os.path.basename(r.path) for r in store.sample_media(10, ab)] == ["y.jpg"]
    assert sorted(os.path.basename(r.path) for r in store.sample_media(10, a)) == [
        "x.jpg",
        "y.jpg",
    ]

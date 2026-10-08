from types import SimpleNamespace

import torch

import app
from app import search, unique_by_path


def test_unique_by_path_keeps_first_of_each_path_in_order():
    items = [("v.mp4", "v 0.9"), ("a.jpg", "a"), ("v.mp4", "v 0.5"), ("b.jpg", "b")]
    assert unique_by_path(items) == [("v.mp4", "v 0.9"), ("a.jpg", "a"), ("b.jpg", "b")]


def test_image_search_returns_each_video_once(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "device", "cpu")
    video, image = str(tmp_path / "v.mp4"), str(tmp_path / "a.jpg")
    for path in (video, image):
        open(path, "wb").close()

    # Two frames of one video plus an image; legacy-style metadata without `frame`.
    collection = SimpleNamespace(
        query=lambda **_: {
            "ids": [["v:0", "v:1", "a"]],
            "documents": [[video, video, image]],
            "embeddings": [[[1.0, 0.0], [0.9, 0.436], [0.8, 0.6]]],
            "metadatas": [[{"media_id": "v"}, {"media_id": "v"}, {"media_id": "a"}]],
        }
    )
    model = SimpleNamespace(
        get_image_features=lambda **_: SimpleNamespace(
            pooler_output=torch.tensor([[1.0, 0.0]])
        )
    )
    processor = lambda **_: SimpleNamespace(to=lambda _device: {})  # noqa: E731

    results = search(
        "", object(), 50, 0.0, 0.0, 0.5, 0.5, model, processor, "siglip",
        1.0, 0.0, SimpleNamespace(get_collection=lambda _: collection),
        str(tmp_path), False,
    )  # fmt: skip

    assert [path for path, _ in results] == [video, image]


def test_search_fetches_requested_number_of_files_not_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "device", "cpu")
    video, a, b = (str(tmp_path / n) for n in ("v.mp4", "a.jpg", "b.jpg"))
    for path in (video, a, b):
        open(path, "wb").close()
    records = [
        ("v:0", video, [1.0, 0.0]),
        ("v:1", video, [0.99, 0.14]),
        ("v:2", video, [0.98, 0.2]),
        ("a", a, [0.9, 0.436]),
        ("b", b, [0.8, 0.6]),
    ]

    def query(n_results, **_):
        ids, docs, embs = zip(*records[:n_results])
        return {
            "ids": [list(ids)],
            "documents": [list(docs)],
            "embeddings": [list(embs)],
            "metadatas": [[{} for _ in ids]],
        }

    model = SimpleNamespace(
        get_image_features=lambda **_: SimpleNamespace(
            pooler_output=torch.tensor([[1.0, 0.0]])
        )
    )
    processor = lambda **_: SimpleNamespace(to=lambda _device: {})  # noqa: E731
    client = SimpleNamespace(get_collection=lambda _: SimpleNamespace(query=query))

    results = search(
        "", object(), 2, 0.0, 0.0, 0.5, 0.5, model, processor, "siglip",
        1.0, 0.0, client, str(tmp_path), False,
    )  # fmt: skip

    assert [path for path, _ in results] == [video, a]

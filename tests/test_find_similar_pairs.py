import chromadb
import numpy as np
import pytest
from PIL import Image

import find_duplicates
from find_duplicates import find_duplicates_in_folder, find_similar_pairs


@pytest.mark.parametrize("col_block_size", [2, 4, 1024])
@pytest.mark.parametrize("threshold", [0.0, 0.9])
def test_find_similar_pairs_matches_brute_force(threshold, col_block_size):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(10, 8)).astype(np.float32)
    # Near-duplicates that straddle block boundaries (block_size=3).
    emb[4] = emb[1] + 0.01
    emb[9] = emb[2] + 0.01
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    paths = [f"img{i}.jpg" for i in range(len(emb))]

    expected = sorted(
        (paths[i], paths[j])
        for i in range(len(emb))
        for j in range(i + 1, len(emb))
        if emb[i] @ emb[j] >= threshold
    )

    pairs = find_similar_pairs(
        emb, paths, threshold, block_size=3, col_block_size=col_block_size
    )

    assert sorted((a, b) for _, a, b in pairs) == expected
    sims = [s for s, _, _ in pairs]
    assert sims == sorted(sims, reverse=True)


def unit(*vecs):
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def pair_scores(emb, owners, threshold=0.9):
    return {(a, b): s for s, a, b in find_similar_pairs(emb, owners, threshold)}


def test_identical_videos_score_one():
    frames = unit([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])
    emb = np.vstack([frames, frames])
    scores = pair_scores(emb, ["a.mp4"] * 3 + ["b.mp4"] * 3)
    assert scores == {("a.mp4", "b.mp4"): pytest.approx(1.0)}


def test_frame_grab_matches_its_video():
    video = unit([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])
    emb = np.vstack([video, video[1:2]])
    scores = pair_scores(emb, ["v.mp4"] * 3 + ["grab.jpg"])
    assert scores == {("v.mp4", "grab.jpg"): pytest.approx(1.0)}


def test_videos_sharing_one_frame_are_not_duplicates():
    shared = [1, 0, 0, 0, 0, 0, 0]
    a = unit(
        shared, [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
    )
    b = unit(
        shared, [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]
    )
    assert pair_scores(np.vstack([a, b]), ["a.mp4"] * 4 + ["b.mp4"] * 4) == {}


def test_frames_of_one_file_never_pair_with_each_other():
    emb = unit([1, 0], [1, 0], [0, 1])
    assert pair_scores(emb, ["v.mp4", "v.mp4", "x.jpg"]) == {}


def test_containment_matches_brute_force_when_many_files_share_a_frame():
    rng = np.random.default_rng(1)
    frames_per_file = [3, 1, 4, 2, 3, 1, 2]
    emb = rng.normal(size=(sum(frames_per_file), 6)).astype(np.float32)
    starts = np.cumsum([0, *frames_per_file[:-1]])
    emb[starts] = emb[0]  # every file shares its first frame
    emb[starts[3] + 1] = emb[1] + 0.01  # and one near-duplicate second frame
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    owners = [f"f{i}" for i, n in enumerate(frames_per_file) for _ in range(n)]

    def contain(a, b):
        sims = (
            emb[[i for i, o in enumerate(owners) if o == a]]
            @ emb[[i for i, o in enumerate(owners) if o == b]].T
        )
        return max(sims.max(axis=1).mean(), sims.max(axis=0).mean())

    files = list(dict.fromkeys(owners))
    expected = {
        (a, b): contain(a, b)
        for i, a in enumerate(files)
        for b in files[i + 1 :]
        if contain(a, b) >= 0.5
    }
    got = pair_scores(emb, owners, threshold=0.5)
    assert got.keys() == expected.keys()
    assert all(got[k] == pytest.approx(expected[k], abs=1e-5) for k in got)


def test_new_files_are_compared_when_saving_their_embeddings_fails(
    tmp_path, monkeypatch
):
    folder = tmp_path / "photos"
    folder.mkdir()
    paths = [str(folder / n) for n in ("a.png", "b.png")]
    for path in paths:
        Image.new("RGB", (4, 4)).save(path)
    monkeypatch.setattr(
        find_duplicates,
        "compute_embeddings",
        lambda image_paths, **_: (unit([1, 0], [1, 0]), list(image_paths)),
    )

    def fail(*_):
        raise RuntimeError("disk full")

    monkeypatch.setattr(find_duplicates, "write_media_embeddings", fail)
    client = chromadb.EphemeralClient()
    client.get_or_create_collection("images", metadata={"hnsw:space": "cosine"})

    pairs = find_duplicates_in_folder(
        str(folder), 0.9, 16, 1024, None, None, client, str(tmp_path / "db"), "cpu"
    )

    assert [(a, b) for _, a, b in pairs] == [tuple(paths)]

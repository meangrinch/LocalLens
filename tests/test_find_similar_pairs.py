import numpy as np
import pytest

from find_duplicates import find_similar_pairs


@pytest.mark.parametrize("threshold", [0.0, 0.9])
def test_find_similar_pairs_matches_brute_force(threshold):
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

    pairs = find_similar_pairs(emb, paths, threshold=threshold, block_size=3)

    assert sorted((a, b) for _, a, b in pairs) == expected
    sims = [s for s, _, _ in pairs]
    assert sims == sorted(sims, reverse=True)

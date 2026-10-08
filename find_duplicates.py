import os
from typing import List, Tuple

import numpy as np
import torch

from index_store import (
    IndexStore,
    build_media_record,
    canonical_path,
    kept_frame_rows,
    path_key,
    scan_media_paths,
    write_media_embeddings,
)
from model_utils import extract_features


def fetch_embeddings_from_db(
    collection, media_ids: List[str], batch_size: int = 1000
) -> dict:
    """Fetch each media's frame embeddings (rows ordered by frame) from ChromaDB."""
    frames: dict = {}
    for i in range(0, len(media_ids), batch_size):
        batch_ids = media_ids[i : i + batch_size]
        try:
            res = collection.get(
                where={"media_id": {"$in": batch_ids}},
                include=["embeddings", "metadatas"],
            )
        except Exception as e:
            print(f"Warning: DB get failed for batch starting at {i}: {e}")
            continue
        for meta, emb in zip(res["metadatas"], res["embeddings"]):
            frames.setdefault(meta["media_id"], []).append(
                (meta.get("frame", 0), np.asarray(emb, dtype=np.float32))
            )
    return {
        media_id: np.stack([emb for _, emb in sorted(rows, key=lambda r: r[0])])
        for media_id, rows in frames.items()
    }


def compute_embeddings(
    image_paths: List[str],
    model,
    processor,
    device: torch.device | str,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    """Compute normalized embeddings in batches; returns one row and path per frame."""
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    all_embeddings: List[np.ndarray] = []
    all_paths: List[str] = []

    total = len(image_paths)
    for batch_index in range(0, total, batch_size):
        batch_paths = image_paths[batch_index : batch_index + batch_size]
        try:
            batch_embeds, processed_paths = extract_features(
                batch_paths,
                model,
                processor,
                device,
            )
        except Exception as e:
            print(f"Error embedding batch starting at {batch_index}: {e}")
            continue

        if processed_paths and batch_embeds.size > 0:
            # Ensure float32 for subsequent similarity math
            all_embeddings.append(batch_embeds.astype(np.float32, copy=False))
            all_paths.extend(processed_paths)

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    embeddings = np.vstack(all_embeddings)
    return embeddings.astype(np.float32, copy=False), all_paths


def find_similar_pairs(
    embeddings: np.ndarray,
    owners: List[str],
    threshold: float,
    block_size: int = 1024,
    device: torch.device | str = "cpu",
    col_block_size: int = 16384,
) -> List[Tuple[float, str, str]]:
    """Find file pairs whose frames match with a containment score >= threshold.

    `embeddings` holds one L2-normalized row per frame and `owners` the file of each
    row. score = max(contain(A, B), contain(B, A)), where contain(A, B) is the mean
    over A's frames of their best cosine against B's frames: trimmed clips and frame
    grabs score high, a single shared frame does not, and two still images score
    their plain cosine. A mean of maxima never exceeds the best frame pair, so
    frame-level hits are a lossless candidate filter.
    """
    files = list(dict.fromkeys(owners))
    if len(files) < 2:
        return []
    file_index = {path: i for i, path in enumerate(files)}
    row_files = [file_index[owner] for owner in owners]
    rows_of: dict = {}
    for row, file_id in enumerate(row_files):
        rows_of.setdefault(file_id, []).append(row)

    # float32 (not half) because half precision is too coarse near thresholds like 0.97
    E = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32)).to(device)
    row_file = torch.tensor(row_files, device=device)
    num_frames = E.shape[0]
    candidates = set()
    for row_start in range(0, num_frames, block_size):
        row_end = min(row_start + block_size, num_frames)
        # Only columns at or after this block: the lower triangle is redundant. Tiling
        # them keeps each similarity block small however large the library is.
        for col_start in range(row_start, num_frames, col_block_size):
            col_end = min(col_start + col_block_size, num_frames)
            sim_block = E[row_start:row_end] @ E[col_start:col_end].T
            rows, cols = torch.nonzero(sim_block >= threshold, as_tuple=True)
            rows, cols = rows + row_start, cols + col_start
            upper = cols > rows
            file_a, file_b = row_file[rows[upper]], row_file[cols[upper]]
            cross = file_a != file_b
            pairs = torch.stack(
                [torch.minimum(file_a, file_b), torch.maximum(file_a, file_b)], dim=1
            )[cross]
            candidates.update(map(tuple, pairs.unique(dim=0).tolist()))

    # Rescore on CPU: these matrices are at most N_FRAMES square, and a GPU sync per
    # pair is ~100x slower when many files share a frame (fades, title cards).
    # TODO: still quadratic in files sharing a frame; score candidates in padded
    # batches if libraries with thousands of such videos get slow.
    frames_np = E.cpu().numpy()
    frames_of = {f: frames_np[rows] for f, rows in rows_of.items()}
    results: List[Tuple[float, str, str]] = []
    for a, b in candidates:
        sims = frames_of[a] @ frames_of[b].T
        score = float(max(sims.max(axis=1).mean(), sims.max(axis=0).mean()))
        if score >= threshold:
            results.append((score, files[a], files[b]))

    results.sort(key=lambda t: t[0], reverse=True)
    return results


def find_duplicates_in_folder(
    folder_path: str,
    threshold: float,
    batch_size: int,
    block_size: int,
    active_model,
    active_processor,
    active_chroma_client,
    db_path: str,
    device: torch.device | str,
) -> List[Tuple[float, str, str]]:
    """Find duplicate pairs in a folder using the active model and ChromaDB."""
    return find_duplicates_in_folders(
        folder_paths=[folder_path],
        threshold=threshold,
        batch_size=batch_size,
        block_size=block_size,
        active_model=active_model,
        active_processor=active_processor,
        active_chroma_client=active_chroma_client,
        db_path=db_path,
        device=device,
    )


def find_duplicates_in_folders(
    folder_paths: List[str],
    threshold: float,
    batch_size: int,
    block_size: int,
    active_model,
    active_processor,
    active_chroma_client,
    db_path: str,
    device: torch.device | str,
) -> List[Tuple[float, str, str]]:
    """Find duplicate pairs across one or more folders."""
    directories = [
        canonical_path(folder_path)
        for folder_path in folder_paths
        if folder_path and os.path.isdir(canonical_path(folder_path))
    ]
    directory_by_key = {path_key(directory): directory for directory in directories}
    directories = sorted(
        directory_by_key.values(),
        key=lambda item: (-len(path_key(item)), path_key(item)),
    )
    if not directories:
        return []

    try:
        collection = active_chroma_client.get_collection(name="images")
    except Exception as e:
        print(f"Warning: Could not get ChromaDB collection: {e}")
        return []

    store = IndexStore(db_path)
    records = []
    seen_paths = set()
    for directory in directories:
        store.upsert_folder(directory)
        for media_path in scan_media_paths(directory):
            record = build_media_record(media_path, directory)
            if record.path_key in seen_paths:
                continue
            seen_paths.add(record.path_key)
            records.append(record)

    if len(records) < 2:
        return []

    media_ids = [record.media_id for record in records]
    frames_by_id = fetch_embeddings_from_db(collection, media_ids, batch_size=1000)
    missing_records = [r for r in records if r.media_id not in frames_by_id]
    if missing_records:
        miss_embeddings, miss_processed = compute_embeddings(
            image_paths=[record.path for record in missing_records],
            model=active_model,
            processor=active_processor,
            device=device,
            batch_size=batch_size,
        )
        if miss_embeddings.size > 0 and len(miss_processed) > 0:
            kept = kept_frame_rows(missing_records, miss_embeddings, miss_processed)
            frames_by_id.update(
                {media_id: miss_embeddings[rows] for media_id, rows in kept.items()}
            )
            try:
                miss_records = write_media_embeddings(
                    collection, missing_records, miss_embeddings, miss_processed
                )
                store.upsert_media_records(miss_records)
            except Exception as e:
                print(f"Warning: failed to upsert embeddings: {e}")

    # One row per frame, in directory order for deterministic output.
    present = [r for r in records if r.media_id in frames_by_id]
    if len(present) < 2:
        return []
    embeddings = np.vstack([frames_by_id[r.media_id] for r in present])
    owners = [r.path for r in present for _ in frames_by_id[r.media_id]]

    # Print processing status in search log style
    print(f"Processing {len(present)} candidates from DB (Duplicates)")

    # Pairwise similarity search
    pairs = find_similar_pairs(
        embeddings,
        owners,
        threshold=threshold,
        block_size=block_size,
        device=device,
    )

    return pairs


def filter_resolved_pairs(pairs: list, trashed_path: str) -> list:
    """Filter out pairs containing a trashed file path."""
    target_key = path_key(trashed_path)
    return [
        p
        for p in pairs
        if path_key(p[1]) != target_key and path_key(p[2]) != target_key
    ]


def _extract_item_path(item) -> str:
    if not item:
        return ""
    if isinstance(item, (list, tuple)):
        return str(item[0]) if item else ""
    if isinstance(item, dict):
        img = item.get("image")
        if isinstance(img, dict):
            return str(img.get("path") or "")
        return str(img or item.get("name") or "")
    return getattr(item, "name", str(item))


def filter_gallery_items(gallery_items: list, trashed_path: str) -> list:
    """Filter out gallery items matching a trashed file path."""
    if not gallery_items:
        return []
    target_key = path_key(trashed_path)
    return [
        item
        for item in gallery_items
        if path_key(_extract_item_path(item)) != target_key
    ]

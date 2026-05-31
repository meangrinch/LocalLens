import os
from typing import List, Tuple

import numpy as np
import torch

from index_store import IndexStore, build_media_record, canonical_path, path_key
from model_utils import extract_features

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def list_image_files(directory_path: str, recursive: bool) -> List[str]:
    """Collect image file paths from a directory."""
    image_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                if any(file_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    image_paths.append(os.path.abspath(os.path.join(root, file_name)))
    else:
        try:
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path) and any(
                    file_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS
                ):
                    image_paths.append(os.path.abspath(file_path))
        except FileNotFoundError:
            return []
    image_paths.sort()
    return image_paths


def fetch_embeddings_from_db(
    collection, ids: List[str], batch_size: int = 1000
) -> dict:
    """Fetch embeddings for given IDs from ChromaDB in batches."""
    id_to_embedding: dict = {}
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        try:
            res = collection.get(ids=batch_ids, include=["embeddings"])
        except Exception as e:
            print(f"Warning: DB get failed for batch starting at {i}: {e}")
            continue
        if not res or not res.get("ids"):
            continue
        for got_id, emb in zip(res["ids"], res.get("embeddings", [])):
            if emb is None:
                continue
            id_to_embedding[got_id] = np.asarray(emb, dtype=np.float32)
    return id_to_embedding


def compute_embeddings(
    image_paths: List[str],
    model,
    processor,
    device: torch.device | str,
    model_type: str,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    """Compute normalized embeddings for images in batches."""
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
                model_type,
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
    paths: List[str],
    threshold: float,
    block_size: int = 1024,
) -> List[Tuple[float, str, str]]:
    """Find pairs of images with cosine similarity >= threshold.

    Assumes embeddings are already L2-normalized. Works in blocks to reduce memory.
    """
    num_images = embeddings.shape[0]
    if num_images < 2:
        return []

    results: List[Tuple[float, str, str]] = []

    # Use float32 for stable dot products
    E = embeddings.astype(np.float32, copy=False)

    # Process by row blocks against the full matrix
    for row_start in range(0, num_images, block_size):
        row_end = min(row_start + block_size, num_images)
        block = E[row_start:row_end]

        # Since vectors are normalized, cosine similarity = dot product
        sim_block = np.matmul(block, E.T)

        # Iterate within the block and only take upper triangle (j > i)
        for local_row, global_i in enumerate(range(row_start, row_end)):
            row_sims = sim_block[local_row]
            j_start = global_i + 1
            if j_start >= num_images:
                continue
            sims = row_sims[j_start:]
            passing = np.where(sims >= threshold)[0]
            for offset in passing.tolist():
                j = j_start + offset
                results.append((float(sims[offset]), paths[global_i], paths[j]))

    results.sort(key=lambda t: t[0], reverse=True)
    return results


def find_duplicates_in_folder(
    folder_path: str,
    threshold: float,
    batch_size: int,
    block_size: int,
    recursive: bool,
    active_model,
    active_processor,
    active_model_type: str,
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
        recursive=recursive,
        active_model=active_model,
        active_processor=active_processor,
        active_model_type=active_model_type,
        active_chroma_client=active_chroma_client,
        db_path=db_path,
        device=device,
    )


def find_duplicates_in_folders(
    folder_paths: List[str],
    threshold: float,
    batch_size: int,
    block_size: int,
    recursive: bool,
    active_model,
    active_processor,
    active_model_type: str,
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

    # Get collection from ChromaDB
    collection = None
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
        for image_path in list_image_files(directory, recursive=recursive):
            record = build_media_record(image_path, directory)
            if record.path_key in seen_paths:
                continue
            seen_paths.add(record.path_key)
            records.append(record)

    if len(records) < 2:
        return []

    record_by_path = {record.path: record for record in records}

    # Try to fetch embeddings from DB
    embeddings_list: List[np.ndarray] = []
    paths_list: List[str] = []
    present_map = {}
    if collection is not None:
        present_map = fetch_embeddings_from_db(
            collection, [record.media_id for record in records], batch_size=1000
        )
    missing_records = [
        record for record in records if record.media_id not in present_map
    ]
    if present_map:
        # Append in directory order for deterministic output
        for record in records:
            if record.media_id in present_map:
                embeddings_list.append(present_map[record.media_id])
                paths_list.append(record.path)

    # Compute missing embeddings
    if missing_records:
        miss_embeddings, miss_processed = compute_embeddings(
            image_paths=[record.path for record in missing_records],
            model=active_model,
            processor=active_processor,
            device=device,
            model_type=active_model_type,
            batch_size=batch_size,
        )
        if miss_embeddings.size > 0 and len(miss_processed) > 0:
            miss_records = [record_by_path[path] for path in miss_processed]
            if collection is not None:
                try:
                    collection.upsert(
                        embeddings=miss_embeddings,
                        documents=[record.path for record in miss_records],
                        ids=[record.media_id for record in miss_records],
                        metadatas=[record.to_metadata() for record in miss_records],
                    )
                except Exception as e:
                    print(f"Warning: failed to upsert embeddings: {e}")
            store.upsert_media_records(miss_records)
            # Append to working arrays following the directory order
            miss_map = {p: e for p, e in zip(miss_processed, miss_embeddings)}
            for record in records:
                if record.path in miss_map:
                    embeddings_list.append(miss_map[record.path])
                    paths_list.append(record.path)

    if len(paths_list) < 2:
        return []

    embeddings = np.vstack(embeddings_list).astype(np.float32, copy=False)
    processed_paths = paths_list

    # Print processing status in search log style
    print(f"Processing {len(processed_paths)} candidates from DB (Duplicates)")

    # Pairwise similarity search
    pairs = find_similar_pairs(
        embeddings,
        processed_paths,
        threshold=threshold,
        block_size=block_size,
    )

    return pairs

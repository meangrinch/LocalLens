import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import chromadb

from model_utils import extract_features, load_model_and_processor


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def get_simplified_model_identifier(model_path_str: str) -> str:
    """Extract simplified model identifier (type+size) from model path.

    Mirrors logic used in the web app to map a model to its DB subfolder.
    """
    name_lower = model_path_str.lower()
    parts = name_lower.split("/")
    model_name_part = parts[-1]

    if "siglip2" in model_name_part:
        sub_parts = model_name_part.split("-")
        if len(sub_parts) >= 2:
            return f"{sub_parts[0]}-{sub_parts[1]}"
        return model_name_part
    elif "clip" in model_name_part:
        sub_parts = model_name_part.split("-")
        if len(sub_parts) >= 3:
            return f"{sub_parts[0]}-{sub_parts[1]}-{sub_parts[2]}"
        return model_name_part
    return model_name_part.replace("-", "_").replace(".", "_")


def default_db_path_for_model(model_path_str: str) -> str:
    return os.path.join("img_db", get_simplified_model_identifier(model_path_str))


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
    return image_paths


def compute_embeddings(
    image_paths: List[str],
    model,
    processor,
    device: str,
    model_type: str,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    """Compute normalized embeddings for images in batches."""
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    all_embeddings: List[np.ndarray] = []
    all_paths: List[str] = []

    total = len(image_paths)
    num_batches = (total + batch_size - 1) // batch_size
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

        processed_so_far = min(batch_index + batch_size, total)
        current_batch_num = (batch_index // batch_size) + 1
        print(
            f"Embeddings: batch {current_batch_num}/{num_batches} (processed {processed_so_far}/{total})"
        )

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    embeddings = np.vstack(all_embeddings)
    return embeddings.astype(np.float32, copy=False), all_paths


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


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate or near-duplicate images using CLIP embeddings."
    )
    parser.add_argument("directory", help="Directory containing images to scan")
    parser.add_argument(
        "--model",
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="Hugging Face model ID or local path (default:laion/CLIP-ViT-H-14-laion2B-s32B-b79K)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to ChromaDB for this model (default: derived from model under img_db)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.93,
        help="Cosine similarity threshold for reporting duplicates (default: 0.995)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (default: 64)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Row block size for pairwise similarity computation (default: 1024)",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Only use embeddings already in the DB; do not compute missing ones",
    )
    parser.add_argument(
        "--no-upsert",
        action="store_true",
        help="Do not upsert newly computed embeddings into the DB",
    )

    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"Error: directory not found: {directory}")
        sys.exit(1)

    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Scanning directory: {directory} (recursive={args.recursive})")
    image_paths = list_image_files(directory, recursive=args.recursive)
    if len(image_paths) < 2:
        print("No images (or only one image) found. Nothing to compare.")
        return
    print(f"Found {len(image_paths)} images to process")

    # Establish DB path and client/collection
    db_path = args.db_path or default_db_path_for_model(args.model)
    os.makedirs(db_path, exist_ok=True)
    client = None
    collection = None
    try:
        client = chromadb.PersistentClient(path=db_path)
        try:
            collection = client.get_collection(name="images")
        except Exception:
            # Create collection if not exists (cosine space)
            collection = client.create_collection(
                name="images", metadata={"hnsw:space": "cosine"}
            )
    except Exception as e:
        print(f"Warning: Could not initialize ChromaDB at {db_path}: {e}")
        if args.db_only:
            print("--db-only specified; aborting because DB is unavailable.")
            return

    # Try to fetch embeddings from DB for the images in the directory
    embeddings_list: List[np.ndarray] = []
    paths_list: List[str] = []
    present_map = {}
    if collection is not None:
        present_map = fetch_embeddings_from_db(collection, image_paths, batch_size=1000)
        if present_map:
            have_cnt = len(present_map)
            print(f"DB: found embeddings for {have_cnt}/{len(image_paths)} images")
    missing_paths = [p for p in image_paths if p not in present_map]
    if present_map:
        # Append in directory order for deterministic output
        for p in image_paths:
            if p in present_map:
                embeddings_list.append(present_map[p])
                paths_list.append(p)

    # If there are missing images, decide whether to compute
    if missing_paths:
        if args.db_only:
            print(
                f"DB-only mode: {len(missing_paths)} images missing embeddings; they will be skipped."
            )
        else:
            print(f"Loading model: {args.model}")
            model, processor, model_type = load_model_and_processor(
                args.model, device, dtype
            )

            t0 = time.time()
            miss_embeddings, miss_processed = compute_embeddings(
                image_paths=missing_paths,
                model=model,
                processor=processor,
                device=device,
                model_type=model_type,
                batch_size=args.batch_size,
            )
            t1 = time.time()
            if miss_embeddings.size == 0 or len(miss_processed) == 0:
                print("No embeddings computed for missing images.")
            else:
                print(
                    f"Computed embeddings for {len(miss_processed)} missing images in {t1 - t0:.2f}s"
                )
                if collection is not None and not args.no_upsert:
                    try:
                        collection.upsert(
                            embeddings=miss_embeddings,
                            documents=miss_processed,
                            ids=miss_processed,
                        )
                        print(
                            f"Upserted {len(miss_processed)} embeddings into DB: {db_path}"
                        )
                    except Exception as e:
                        print(f"Warning: failed to upsert embeddings: {e}")
                # Append to working arrays following the directory order
                miss_map = {p: e for p, e in zip(miss_processed, miss_embeddings)}
                for p in image_paths:
                    if p in miss_map:
                        embeddings_list.append(miss_map[p])
                        paths_list.append(p)

    if len(paths_list) < 2:
        print("Not enough images with embeddings to compare.")
        return
    embeddings = np.vstack(embeddings_list).astype(np.float32, copy=False)
    processed_paths = paths_list

    # Pairwise similarity search
    print(f"Finding pairs with cosine similarity >= {args.threshold}")
    pairs = find_similar_pairs(
        embeddings,
        processed_paths,
        threshold=args.threshold,
        block_size=args.block_size,
    )

    if not pairs:
        print("No duplicate or near-duplicate pairs found.")
        return

    print(f"Found {len(pairs)} pairs:")
    for sim, p1, p2 in pairs:
        print(f"{sim:.6f}\t{p1}\t{p2}")


if __name__ == "__main__":
    main()

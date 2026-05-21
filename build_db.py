import argparse
import os
import time

import chromadb
import torch

from device import get_best_device, get_best_dtype
from index_store import (
    IndexStore,
    batched,
    build_media_records,
    canonical_path,
    migrate_legacy_db,
    path_key,
    write_indexed_folders_mirror,
)
from model_utils import extract_features, load_model_and_processor

torch.set_float32_matmul_precision("high")

model = None
processor = None
model_type = None
client = None
device = get_best_device()
dtype = get_best_dtype(device)


def _emit_progress(progress_callback, status: str, **kwargs) -> None:
    if progress_callback:
        progress_callback(status=status, **kwargs)


def _delete_chroma_ids(collection, media_ids: list[str]) -> None:
    for batch in batched(media_ids, 500):
        try:
            collection.delete(ids=batch)
        except Exception as e:
            print(f"Error deleting batch: {e}")


def _store_processed_records(db_path_str: str, records) -> None:
    IndexStore(db_path_str).upsert_media_records(list(records))


def _upsert_records_with_store(
    db_path_str: str,
    records,
    collection,
    model_param,
    processor_param,
    device,
    model_type_param_ext,
    batch_size_param: int,
    progress_callback=None,
) -> int:
    if not records:
        return 0

    record_by_key = {record.path_key: record for record in records}
    processed_records = []
    processed_count = 0
    total_batches = (len(records) + batch_size_param - 1) // batch_size_param

    for batch_number, batch_records in enumerate(
        batched(records, batch_size_param), start=1
    ):
        batch_paths = [record.path for record in batch_records]
        try:
            batch_embeddings, processed_files = extract_features(
                batch_paths,
                model_param,
                processor_param,
                device,
                model_type_param_ext,
            )
            if not processed_files:
                print(f"Batch failed: {len(batch_paths)} files could not be processed")
                continue

            batch_processed_records = [
                record_by_key[path_key(path)] for path in processed_files
            ]
            collection.upsert(
                embeddings=batch_embeddings,
                documents=[record.path for record in batch_processed_records],
                ids=[record.media_id for record in batch_processed_records],
                metadatas=[record.to_metadata() for record in batch_processed_records],
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

        processed_records.extend(batch_processed_records)
        processed_count += len(batch_processed_records)
        print(
            f"Batch {batch_number}/{total_batches}: {len(batch_processed_records)} images done."
        )
        _emit_progress(
            progress_callback,
            "batch_processed",
            current_batch_num=batch_number,
            total_batches=total_batches,
            images_in_batch=len(batch_processed_records),
            cumulative_processed_this_run=processed_count,
            total_images_to_process=len(records),
        )

    _store_processed_records(db_path_str, processed_records)
    return processed_count


def process_images(
    folders,
    collection,
    mode="add",
    model_param=None,
    processor_param=None,
    device=None,
    model_type_param_ext=None,
    progress_callback=None,
    batch_size_param=128,
    db_path_str: str | None = None,
):
    if not db_path_str:
        raise ValueError("db_path_str is required for indexed processing.")

    migrate_legacy_db(db_path_str, collection)
    store = IndexStore(db_path_str)

    if mode == "add":
        print(f"Adding folders: {', '.join(folders)}")
        records_to_embed = []
        for folder_arg in folders:
            folder_path = canonical_path(folder_arg)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder not found: {folder_path}")
                continue
            store.upsert_folder(folder_path)
            folder_records = build_media_records(folder_path)
            records_to_embed.extend(store.records_needing_embedding(folder_records))

        _emit_progress(
            progress_callback,
            "start_processing",
            folders_being_processed=folders,
            total_images_to_process=len(records_to_embed),
        )

        if not records_to_embed:
            print("No new or changed media files found to add")
            _emit_progress(
                progress_callback, "all_batches_done", total_successfully_added=0
            )
            return

        print(f"Processing {len(records_to_embed)} images...")
        processed_count = _upsert_records_with_store(
            db_path_str,
            records_to_embed,
            collection,
            model_param,
            processor_param,
            device,
            model_type_param_ext,
            batch_size_param,
            progress_callback,
        )
        print(f"Successfully added or updated {processed_count} images")
        _emit_progress(
            progress_callback,
            "all_batches_done",
            total_successfully_added=processed_count,
        )

    elif mode == "update":
        print("Updating database...")
        total_added = 0
        total_deleted = 0
        for folder_arg in folders:
            folder_path = canonical_path(folder_arg)
            store.upsert_folder(folder_path)
            if not os.path.isdir(folder_path):
                print(f"Warning: Indexed folder not found: {folder_path}")
                continue

            filesystem_records = build_media_records(folder_path)
            filesystem_ids = {record.media_id for record in filesystem_records}
            stored_records = store.get_media_by_folder(folder_path)
            stored_ids = {record.media_id for record in stored_records}
            stale_ids = sorted(stored_ids - filesystem_ids)
            if stale_ids:
                print(f"Deleting {len(stale_ids)} stale images...")
                _delete_chroma_ids(collection, stale_ids)
                store.delete_media_ids(stale_ids)
                total_deleted += len(stale_ids)

            records_to_embed = store.records_needing_embedding(filesystem_records)
            if records_to_embed:
                print(f"Adding {len(records_to_embed)} images...")
                total_added += _upsert_records_with_store(
                    db_path_str,
                    records_to_embed,
                    collection,
                    model_param,
                    processor_param,
                    device,
                    model_type_param_ext,
                    batch_size_param,
                )

        if total_added == 0 and total_deleted == 0:
            print("Database is up to date")
        else:
            print(f"Update complete: {total_added} added, {total_deleted} deleted")


def db_add_folders(
    folders_to_process: list[str],
    db_path_str: str,
    collection_obj,
    model_obj,
    processor_obj,
    device: torch.device,
    model_type_str: str,
    batch_size: int,
    progress_callback=None,
):
    start_time = time.time()
    migrate_legacy_db(db_path_str, collection_obj)
    store = IndexStore(db_path_str)
    before_folders = set(store.list_folders())

    process_images(
        folders_to_process,
        collection_obj,
        mode="add",
        model_param=model_obj,
        processor_param=processor_obj,
        device=device,
        model_type_param_ext=model_type_str,
        progress_callback=progress_callback,
        batch_size_param=batch_size,
        db_path_str=db_path_str,
    )

    folders = store.list_folders()
    write_indexed_folders_mirror(db_path_str, folders)
    added_folders = set(folders) - before_folders
    if added_folders:
        print(f"Added {len(added_folders)} folders to index")
    else:
        print("Folders already indexed")

    end_time = time.time()
    print(f"Add folder completed in {(end_time - start_time):.2f}s")
    _emit_progress(
        progress_callback,
        "add_folder_completed",
        duration_seconds=(end_time - start_time),
    )


def db_update_indexed_folders(
    db_path_str: str,
    collection_obj,
    model_obj,
    processor_obj,
    device: torch.device,
    model_type_str: str,
    batch_size: int,
):
    start_time = time.time()
    migrate_legacy_db(db_path_str, collection_obj)
    store = IndexStore(db_path_str)
    indexed_folders = store.list_folders()
    if indexed_folders:
        process_images(
            indexed_folders,
            collection_obj,
            mode="update",
            model_param=model_obj,
            processor_param=processor_obj,
            device=device,
            model_type_param_ext=model_type_str,
            batch_size_param=batch_size,
            db_path_str=db_path_str,
        )
        write_indexed_folders_mirror(db_path_str, store.list_folders())
    else:
        print("Index empty - nothing to update")
    end_time = time.time()
    print(f"Update completed in {(end_time - start_time):.2f}s")


def db_delete_folder(
    folder_to_delete_str: str, db_path_str: str, collection_obj
) -> bool:
    start_time = time.time()
    migrate_legacy_db(db_path_str, collection_obj)
    store = IndexStore(db_path_str)
    action_taken, media_ids = store.delete_folder(folder_to_delete_str)
    if media_ids:
        _delete_chroma_ids(collection_obj, media_ids)
        print(f"Successfully removed {len(media_ids)} images from database")
    if action_taken:
        print(f"Removed folder from index: {os.path.basename(folder_to_delete_str)}")
    else:
        print(f"Folder not found in index: {folder_to_delete_str}")
    write_indexed_folders_mirror(db_path_str, store.list_folders())
    end_time = time.time()
    print(f"Delete folder completed in {(end_time - start_time):.2f}s")
    return action_taken


def ensure_file_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.isfile(filepath):
        open(filepath, "a").close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index image folders for a specific model and database."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Hugging Face ID or local path (e.g., 'google/siglip2-so400m-patch16-512')",
    )
    parser.add_argument(
        "--db_path",
        required=True,
        help="Path to the ChromaDB directory for this model (e.g., 'img_db/siglip2_so400m')",
    )
    parser.add_argument(
        "--add",
        nargs="+",
        help="Folders to add to the database. Creates DB if not exists.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing database by rescanning indexed folders.",
    )
    parser.add_argument(
        "--delete_folder",
        type=str,
        help="Folder path to remove from the index and database.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="Batch size for adding images to DB."
    )
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)

    model, processor, model_type = load_model_and_processor(
        args.model_path, device, dtype
    )
    print(f"Model loaded: {model_type.upper()}")

    print(f"Initializing ChromaDB: {args.db_path}")
    client = chromadb.PersistentClient(path=args.db_path)

    collection_name = "images"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: '{collection_name}'")
    except Exception:
        collection = client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(f"Created collection: '{collection_name}'")

    migrate_legacy_db(args.db_path, collection)

    if args.add:
        db_add_folders(
            folders_to_process=args.add,
            db_path_str=args.db_path,
            collection_obj=collection,
            model_obj=model,
            processor_obj=processor,
            device=device,
            model_type_str=model_type,
            batch_size=args.batch_size,
        )
    elif args.update:
        db_update_indexed_folders(
            db_path_str=args.db_path,
            collection_obj=collection,
            model_obj=model,
            processor_obj=processor,
            device=device,
            model_type_str=model_type,
            batch_size=args.batch_size,
        )
    elif args.delete_folder:
        db_delete_folder(
            folder_to_delete_str=args.delete_folder,
            db_path_str=args.db_path,
            collection_obj=collection,
        )
    else:
        print("No action specified (--add, --update, or --delete_folder)")

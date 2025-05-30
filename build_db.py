import argparse
import os
import time

import chromadb
import torch

from model_utils import extract_features, load_model_and_processor

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]

torch.set_float32_matmul_precision("high")

# Global variables for model and client, to be initialized in main
model = None
processor = None
model_type = None
client = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def extract_features_clip(image_paths_batch, model_param, processor_param, device_param, model_type_param):
    """
    Extracts CLIP features for a batch of image paths.
    Skips images that cannot be opened and prints a warning.
    Returns a tuple: (list_of_embeddings, list_of_successfully_processed_paths)
    """

    return extract_features(
        image_paths_batch, model_param, processor_param, device_param, model_type_param
    )


def process_images(
    folders,
    collection,
    mode="add",
    model_param=None,
    processor_param=None,
    device_param_ext=None,
    model_type_param_ext=None,
):
    if mode == "add":
        print(f"Adding folders: {', '.join(folders)}")
        all_files_to_add = []
        for folder_arg in folders:
            folder_path = os.path.abspath(folder_arg)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder not found: {folder_path}")
                continue
            for f_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, f_name)
                if os.path.isfile(file_path) and any(f_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    all_files_to_add.append(os.path.abspath(file_path))

        if not all_files_to_add:
            print("No image files found to add")
        else:
            print(f"Processing {len(all_files_to_add)} images...")
            batch_size = 128
            processed_count = 0
            for i in range(0, len(all_files_to_add), batch_size):
                batch_files = all_files_to_add[i: i + batch_size]
                try:
                    batch_embeddings, processed_files = extract_features_clip(
                        batch_files,
                        model_param,
                        processor_param,
                        device_param_ext,
                        model_type_param_ext,
                    )
                    if processed_files:
                        collection.add(
                            embeddings=batch_embeddings,
                            documents=processed_files,
                            ids=processed_files,
                        )
                        processed_count += len(processed_files)
                    elif batch_files:
                        print(f"Batch failed: {len(batch_files)} files could not be processed")
                except Exception as e:
                    print(f"Error processing batch: {e}")
            if processed_count > 0:
                print(f"Successfully added {processed_count} images")

    elif mode == "update":
        print("Updating database...")

        current_filesystem_files = set()
        for indexed_folder_path_arg in folders:
            indexed_folder_path = os.path.abspath(indexed_folder_path_arg)
            if not os.path.isdir(indexed_folder_path):
                print(f"Warning: Indexed folder not found: {indexed_folder_path}")
                continue
            for f_name in os.listdir(indexed_folder_path):
                file_path = os.path.join(indexed_folder_path, f_name)
                if os.path.isfile(file_path) and any(f_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    current_filesystem_files.add(os.path.abspath(file_path))

        existing_db_files = set()
        try:
            db_content = collection.get(include=["documents"])
            if db_content and db_content["documents"] is not None:
                existing_db_files = set(os.path.abspath(doc) for doc in db_content["documents"])
        except Exception as e:
            print(f"Error fetching from database: {e}")
            return

        files_to_add = list(current_filesystem_files - existing_db_files)
        files_to_delete = list(existing_db_files - current_filesystem_files)

        print(f"Found {len(current_filesystem_files)} files in filesystem, {len(existing_db_files)} in database")

        if files_to_delete:
            print(f"Deleting {len(files_to_delete)} stale images...")
            delete_batch_size = 500
            for i in range(0, len(files_to_delete), delete_batch_size):
                batch_delete_ids = files_to_delete[i: i + delete_batch_size]
                try:
                    collection.delete(ids=batch_delete_ids)
                except Exception as e:
                    print(f"Error deleting batch: {e}")

        if files_to_add:
            print(f"Adding {len(files_to_add)} new images...")
            add_batch_size = 128
            processed_count = 0
            for i in range(0, len(files_to_add), add_batch_size):
                batch_add_files = files_to_add[i: i + add_batch_size]
                try:
                    batch_embeddings, processed_files = extract_features_clip(
                        batch_add_files,
                        model_param,
                        processor_param,
                        device_param_ext,
                        model_type_param_ext,
                    )
                    if processed_files:
                        collection.upsert(
                            embeddings=batch_embeddings,
                            documents=processed_files,
                            ids=processed_files,
                        )
                        processed_count += len(processed_files)
                    elif batch_add_files:
                        print(f"Batch failed: {len(batch_add_files)} files could not be processed")
                except Exception as e:
                    print(f"Error processing batch: {e}")
            if processed_count > 0:
                print(f"Successfully added {processed_count} new images")
        else:
            print("Database is up to date")


def db_add_folders(
    folders_to_process: list[str],
    db_path_str: str,
    collection_obj,
    model_obj,
    processor_obj,
    device_str: str,
    model_type_str: str,
):
    """Adds specified folders to the database and updates the index file."""
    start_time = time.time()
    indexed_folders_file_path = os.path.join(db_path_str, "indexed_folders.txt")
    ensure_file_exists(indexed_folders_file_path)

    current_indexed_folders = set()
    if os.path.exists(indexed_folders_file_path) and os.path.getsize(indexed_folders_file_path) > 0:
        with open(indexed_folders_file_path, "r") as f:
            current_indexed_folders = set(line.strip() for line in f if line.strip())

    folders_to_actually_add_to_txt = [f for f in folders_to_process if f not in current_indexed_folders]

    process_images(
        folders_to_process,
        collection_obj,
        mode="add",
        model_param=model_obj,
        processor_param=processor_obj,
        device_param_ext=device_str,
        model_type_param_ext=model_type_str,
    )

    if folders_to_actually_add_to_txt:
        with open(indexed_folders_file_path, "a") as f:
            if current_indexed_folders:
                f.write("\n")
            f.write("\n".join(folders_to_actually_add_to_txt))
        print(f"Added {len(folders_to_actually_add_to_txt)} folders to index file")
    else:
        print("Folders already indexed")
    end_time = time.time()
    print(f"Add folders completed in {(end_time - start_time):.2f}s")


def db_update_indexed_folders(
    db_path_str: str,
    collection_obj,
    model_obj,
    processor_obj,
    device_str: str,
    model_type_str: str,
):
    """Updates the database by rescanning folders listed in the index file."""
    start_time = time.time()
    indexed_folders_file_path = os.path.join(db_path_str, "indexed_folders.txt")
    ensure_file_exists(indexed_folders_file_path)

    if os.path.exists(indexed_folders_file_path) and os.path.getsize(indexed_folders_file_path) > 0:
        with open(indexed_folders_file_path, "r") as f:
            indexed_folders = [line.strip() for line in f if line.strip()]
        if indexed_folders:
            process_images(
                indexed_folders,
                collection_obj,
                mode="update",
                model_param=model_obj,
                processor_param=processor_obj,
                device_param_ext=device_str,
                model_type_param_ext=model_type_str,
            )
        else:
            print("No folders in index file to update")
    else:
        print("Index file empty - nothing to update")
    end_time = time.time()
    print(f"Update completed in {(end_time - start_time):.2f}s")


def db_delete_folder(folder_to_delete_str: str, db_path_str: str, collection_obj):
    """Deletes a folder from the index file and removes its images from the database."""
    start_time = time.time()
    folder_to_delete_abs = os.path.abspath(folder_to_delete_str)
    indexed_folders_file_path = os.path.join(db_path_str, "indexed_folders.txt")
    ensure_file_exists(indexed_folders_file_path)

    # Remove from indexed_folders.txt
    raw_indexed_folders_read = []
    if os.path.exists(indexed_folders_file_path) and os.path.getsize(indexed_folders_file_path) > 0:
        with open(indexed_folders_file_path, "r") as f:
            raw_indexed_folders_read = [line.strip() for line in f if line.strip()]

    final_folders_to_write = []
    folder_was_found_for_removal = False
    for raw_path in raw_indexed_folders_read:
        if os.path.abspath(raw_path) == folder_to_delete_abs:
            folder_was_found_for_removal = True
        else:
            final_folders_to_write.append(raw_path)

    if folder_was_found_for_removal:
        with open(indexed_folders_file_path, "w") as f:
            f.write("\n".join(final_folders_to_write))
        print(f"Removed folder from index: {os.path.basename(folder_to_delete_abs)}")
    else:
        print(f"Folder not found in index: {os.path.basename(folder_to_delete_abs)}")

    # Remove images from ChromaDB
    if not os.path.isdir(folder_to_delete_abs):
        print(f"Warning: Folder not found on disk: {folder_to_delete_abs}")
    else:
        files_to_remove_from_db = []
        for root, _, files in os.walk(folder_to_delete_abs):
            for f_name in files:
                if any(f_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    files_to_remove_from_db.append(os.path.abspath(os.path.join(root, f_name)))

        if files_to_remove_from_db:
            try:
                collection_obj.delete(ids=files_to_remove_from_db)
                print(f"Successfully removed {len(files_to_remove_from_db)} images from database")
            except Exception as e:
                print(f"Error removing images from database: {e}")
        else:
            print("No images found to remove")
    end_time = time.time()
    print(f"Delete folder completed in {(end_time - start_time):.2f}s")


def ensure_file_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.isfile(filepath):
        open(filepath, "a").close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index image folders for a specific model and database.")
    parser.add_argument(
        "--model_path", required=True, help="Path to the model (e.g., 'google/siglip2-so400m-patch16-512')"
    )
    parser.add_argument(
        "--db_path",
        required=True,
        help="Path to the ChromaDB directory for this model (e.g., 'img_db_google_siglip2_so400m_patch16_512/')",
    )
    parser.add_argument("--add", nargs="+", help="Folders to add to the database. Creates DB if not exists.")
    parser.add_argument("--update", action="store_true", help="Update existing database by rescanning indexed folders.")
    parser.add_argument("--delete_folder", type=str, help="Folder path to remove from the index and database.")
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model, processor, model_type = load_model_and_processor(args.model_path, device, dtype)
    print(f"Model loaded: {model_type.upper()}")

    print(f"Initializing ChromaDB: {args.db_path}")
    client = chromadb.PersistentClient(path=args.db_path)

    indexed_folders_file_path = os.path.join(args.db_path, "indexed_folders.txt")
    ensure_file_exists(indexed_folders_file_path)
    collection_name = "images"

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: '{collection_name}'")
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        print(f"Created collection: '{collection_name}'")

    if args.add:
        db_add_folders(
            folders_to_process=args.add,
            db_path_str=args.db_path,
            collection_obj=collection,
            model_obj=model,
            processor_obj=processor,
            device_str=device,
            model_type_str=model_type
        )
    elif args.update:
        db_update_indexed_folders(
            db_path_str=args.db_path,
            collection_obj=collection,
            model_obj=model,
            processor_obj=processor,
            device_str=device,
            model_type_str=model_type
        )
    elif args.delete_folder:
        db_delete_folder(
            folder_to_delete_str=args.delete_folder,
            db_path_str=args.db_path,
            collection_obj=collection
        )
    else:
        print("No action specified (--add, --update, or --delete_folder)")

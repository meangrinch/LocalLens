import asyncio
import sys

# Silence "WinError 10054" console spam
if sys.platform == "win32":
    if hasattr(asyncio, "proactor_events"):
        _ProactorBasePipeTransport = asyncio.proactor_events._ProactorBasePipeTransport
    else:
        from asyncio.proactor_events import _ProactorBasePipeTransport

    _original_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def _safe_connection_lost(self, exc=None):
        try:
            _original_connection_lost(self, exc)
        except ConnectionResetError:
            pass
        except OSError as e:
            if e.winerror == 10054:
                pass
            else:
                raise

    _ProactorBasePipeTransport._call_connection_lost = _safe_connection_lost

import os
import random
import time

import chromadb
import gradio as gr
import torch
import torch.nn.functional as F

from build_db import db_add_folders, db_delete_folder, db_update_indexed_folders
from find_duplicates import find_duplicates_in_folder
from model_utils import load_model_and_processor

__version__ = "1.2.3"

# --- Configuration ---
AVAILABLE_MODELS = [
    "google/siglip2-giant-opt-patch16-384",
    "google/siglip2-so400m-patch16-512",
    "apple/DFN5B-CLIP-ViT-H-14-378",
    "facebook/metaclip-h14-fullcc2.5b",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "openai/clip-vit-large-patch14",
]
DEFAULT_MODEL_PATH = None

INITIAL_N_RESULTS = 50  # Number of results to fetch from ChromaDB before filtering

FALLBACK_SIGLIP_THRESHOLD = -8.0
FALLBACK_CLIP_THRESHOLD = 20.0
FALLBACK_COMBINED_THRESHOLD = 0.45
FALLBACK_IMAGE_ONLY_THRESHOLD = 0.70
MODEL_CONFIDENCE_DEFAULTS = {
    "google/siglip2-giant-opt-patch16-384": {
        "siglip_thresh": -8.0,
        "combined_thresh": 0.50,
        "image_only_thresh": 0.85,
    },
    "google/siglip2-so400m-patch16-512": {
        "siglip_thresh": -8.0,
        "combined_thresh": 0.50,
        "image_only_thresh": 0.85,
    },
    "apple/DFN5B-CLIP-ViT-H-14-378": {
        "clip_thresh": 3.5,
        "combined_thresh": 0.45,
        "image_only_thresh": 0.70,
    },
    "facebook/metaclip-h14-fullcc2.5b": {
        "clip_thresh": 31.0,
        "combined_thresh": 0.45,
        "image_only_thresh": 0.70,
    },
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "clip_thresh": 24.0,
        "combined_thresh": 0.45,
        "image_only_thresh": 0.70,
    },
    "openai/clip-vit-large-patch14": {
        "clip_thresh": 22.0,
        "combined_thresh": 0.45,
        "image_only_thresh": 0.70,
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def get_simplified_model_identifier(model_path_str: str) -> str:
    """Extracts a simplified model identifier (type+size) from the model path."""
    name_lower = model_path_str.lower()
    parts = name_lower.split("/")
    model_name_part = parts[-1]

    if "siglip2" in model_name_part:
        sub_parts = model_name_part.split("-")
        if len(sub_parts) >= 2:
            return f"{sub_parts[0]}-{sub_parts[1]}"  # "siglip2-so400m"
        return model_name_part
    elif "clip" in model_name_part:
        sub_parts = model_name_part.split("-")
        if len(sub_parts) >= 3:
            return f"{sub_parts[0]}-{sub_parts[1]}-{sub_parts[2]}"  # "clip-vit-large"
        return model_name_part
    return model_name_part.replace("-", "_").replace(".", "_")


def generate_db_path_for_model(model_path_str: str) -> str:
    """Generates a database directory path like 'img_db/simplified_model_id/'."""
    simplified_id = get_simplified_model_identifier(model_path_str)
    return os.path.join("img_db", simplified_id)


def read_indexed_folders(db_path: str) -> list[str]:
    """Reads the indexed_folders.txt file from the given db_path and returns a list of folder paths."""
    if not db_path:
        return []
    indexed_folders_file = os.path.join(db_path, "indexed_folders.txt")
    folders = []
    if os.path.exists(indexed_folders_file):
        try:
            with open(indexed_folders_file, "r") as f:
                folders = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading indexed folders: {e}")
            gr.Warning(f"Could not read indexed folders file: {indexed_folders_file}")
    return folders


def load_and_switch_model_db(
    selected_model_path: str, progress=gr.Progress(track_tqdm=True)
):
    """Loads the selected model, its processor, and initializes/loads its specific ChromaDB."""
    yield (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        gr.Textbox(value="Loading model...", interactive=False),
        None,
        None,
        None,
        None,
        gr.Dropdown(interactive=False),
    )

    if not selected_model_path:
        yield (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "No model selected.",
            FALLBACK_SIGLIP_THRESHOLD,
            FALLBACK_CLIP_THRESHOLD,
            FALLBACK_COMBINED_THRESHOLD,
            FALLBACK_IMAGE_ONLY_THRESHOLD,
            gr.Dropdown(interactive=True),
        )
        return

    progress(0, desc="Starting model...")
    ui_siglip_thresh_to_set = FALLBACK_SIGLIP_THRESHOLD
    ui_clip_thresh_to_set = FALLBACK_CLIP_THRESHOLD
    ui_combined_thresh_to_set = FALLBACK_COMBINED_THRESHOLD
    ui_image_only_thresh_to_set = FALLBACK_IMAGE_ONLY_THRESHOLD
    new_db_path = generate_db_path_for_model(selected_model_path)
    progress(0.1, desc=f"DB path: {new_db_path}")
    if not os.path.exists(new_db_path):
        os.makedirs(new_db_path)
        progress(0.15, desc=f"Created directory: {new_db_path}")
    indexed_folders_file = os.path.join(new_db_path, "indexed_folders.txt")
    if not os.path.exists(indexed_folders_file):
        try:
            if not os.path.exists(indexed_folders_file):
                with open(indexed_folders_file, "a"):
                    pass
                progress(0.2, desc=f"Created {indexed_folders_file}")
        except Exception as e:
            print(f"Error creating indexed folders file: {e}")
    try:
        progress(0.3, desc="Loading model...")
        new_model, new_processor, new_model_type = load_model_and_processor(
            selected_model_path, device, dtype
        )
        progress(0.6, desc="Calculating params...")
        new_logit_scale_exp_val = (
            new_model.logit_scale.exp().item()
            if hasattr(new_model, "logit_scale")
            else None
        )
        new_logit_bias_val = (
            new_model.logit_bias.item() if hasattr(new_model, "logit_bias") else None
        )
        model_specific_conf = MODEL_CONFIDENCE_DEFAULTS.get(selected_model_path, {})
        if new_model_type == "siglip":
            ui_siglip_thresh_to_set = model_specific_conf.get(
                "siglip_thresh", FALLBACK_SIGLIP_THRESHOLD
            )
        elif new_model_type == "clip":
            ui_clip_thresh_to_set = model_specific_conf.get(
                "clip_thresh", FALLBACK_CLIP_THRESHOLD
            )
        ui_combined_thresh_to_set = model_specific_conf.get(
            "combined_thresh", FALLBACK_COMBINED_THRESHOLD
        )
        ui_image_only_thresh_to_set = model_specific_conf.get(
            "image_only_thresh", FALLBACK_IMAGE_ONLY_THRESHOLD
        )
        progress(0.7, desc="Initializing ChromaDB client...")
        new_chroma_client = chromadb.PersistentClient(path=new_db_path)
        try:
            new_chroma_client.get_or_create_collection(
                name="images", metadata={"hnsw:space": "cosine"}
            )
        except Exception as db_e:
            print(f"ChromaDB collection error: {db_e}")
            gr.Warning(
                f"ChromaDB client initialized for '{new_db_path}', but error ensuring collection: {db_e}"
            )
        progress(0.9, desc="ChromaDB client initialized.")
        gr.Info(
            f"Successfully switched to model: {selected_model_path} and DB: {new_db_path}"
        )
        current_indexed_folders = read_indexed_folders(new_db_path)
        yield (
            selected_model_path,
            new_db_path,
            new_model,
            new_processor,
            new_model_type,
            new_logit_scale_exp_val,
            new_logit_bias_val,
            new_chroma_client,
            "\n".join(current_indexed_folders),
            ui_siglip_thresh_to_set,
            ui_clip_thresh_to_set,
            ui_combined_thresh_to_set,
            ui_image_only_thresh_to_set,
            gr.Dropdown(interactive=True),
        )
    except Exception as e:
        print(f"Error loading model/DB: {e}")
        gr.Error(f"Error loading model/DB for {selected_model_path}: {e}")
        current_indexed_folders_on_error = read_indexed_folders(new_db_path)
        yield (
            selected_model_path,
            new_db_path,
            None,
            None,
            None,
            None,
            None,
            None,
            "\n".join(current_indexed_folders_on_error),
            FALLBACK_SIGLIP_THRESHOLD,
            FALLBACK_CLIP_THRESHOLD,
            FALLBACK_COMBINED_THRESHOLD,
            FALLBACK_IMAGE_ONLY_THRESHOLD,
            gr.Dropdown(interactive=True),
        )


def handle_update_sync_button_click(
    current_model_path: str,
    current_db_path: str,
    active_model_state_val,
    active_processor_state_val,
    active_model_type_state_val,
    active_chroma_client_state_val,
    batch_size_ui: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Calls db_update_indexed_folders directly to update/sync the active database."""
    if not current_model_path or not current_db_path:
        gr.Warning("No active model or DB path. Please select a model first.")
        return
    if (
        not active_model_state_val
        or not active_processor_state_val
        or not active_model_type_state_val
        or not active_chroma_client_state_val
    ):
        gr.Warning(
            "Model components (model, processor, type, or DB client) not loaded. Please re-select a model."
        )
        return

    progress(0, desc="Starting DB update...")

    try:
        db_update_indexed_folders(
            db_path_str=current_db_path,
            collection_obj=active_chroma_client_state_val.get_collection("images"),
            model_obj=active_model_state_val,
            processor_obj=active_processor_state_val,
            device_str=device,
            model_type_str=active_model_type_state_val,
            batch_size=batch_size_ui,
        )
        gr.Info(f"Database '{current_db_path}' update process completed successfully.")
        progress(1, desc="DB updated.")

    except Exception as e:
        print(f"Error updating DB directly: {e}")
        gr.Error(
            f"An unexpected error occurred while trying to update DB '{current_db_path}': {e}"
        )
        progress(1, desc="Error.")


def handle_add_folder_button_click(
    new_folder_to_add: str,
    current_model_path: str,
    current_db_path: str,
    active_model_state_val,
    active_processor_state_val,
    active_model_type_state_val,
    active_chroma_client_state_val,
    batch_size_ui: int,
    progress=gr.Progress(),
):
    """Calls db_add_folders directly to add a new folder to the active DB."""
    new_folder_stripped = new_folder_to_add.strip() if new_folder_to_add else ""

    if not new_folder_stripped:
        gr.Warning("Folder path cannot be empty.")
        return "\n".join(read_indexed_folders(current_db_path))

    if not current_model_path or not current_db_path:
        gr.Warning("No active model or DB path. Please select a model first.")
        return "\n".join(read_indexed_folders(current_db_path))

    if not os.path.isdir(new_folder_stripped):
        print(f"Folder not found on disk: {os.path.abspath(new_folder_stripped)}")
        gr.Warning(f"Folder not found: {new_folder_stripped}")
        return "\n".join(read_indexed_folders(current_db_path))

    def update_gradio_progress(status, **kwargs):
        if status == "start_processing":
            total_images = kwargs.get("total_images_to_process", 0)
            folders_being_processed = kwargs.get("folders_being_processed", [])
            if folders_being_processed:
                folder_names_str = ", ".join(
                    [os.path.basename(f) for f in folders_being_processed]
                )
            else:
                folder_names_str = "selected folder(s)"
            if total_images > 0:
                progress(
                    0,
                    desc=f"Processing {total_images} images from {folder_names_str}...",
                )
            else:
                progress(0, desc=f"No images to process in {folder_names_str}.")
        elif status == "batch_processed":
            current_batch = kwargs.get("current_batch_num", 0)
            total_batches = kwargs.get("total_batches", 0)
            images_in_batch = kwargs.get("images_in_batch", 0)
            cumulative_processed = kwargs.get("cumulative_processed_this_run", 0)
            total_images_to_process = kwargs.get("total_images_to_process", 1)

            prog_fraction = (
                cumulative_processed / total_images_to_process
                if total_images_to_process > 0
                else 0
            )
            desc_str = (
                f"Batch {current_batch}/{total_batches} ({images_in_batch} images). "
                f"Total: {cumulative_processed}/{total_images_to_process}"
            )
            progress(prog_fraction, desc=desc_str)
        elif status == "all_batches_done":
            total_added = kwargs.get("total_successfully_added", 0)
            progress(1, desc=f"Successfully processed {total_added} images.")
        elif status == "add_folder_completed":
            duration = kwargs.get("duration_seconds", 0)
            gr.Info(
                f"Folder '{os.path.basename(new_folder_stripped)}' processed in {duration:.2f}s."
            )
            progress(1, desc=f"Add folder completed in {duration:.2f}s.")

    try:
        db_add_folders(
            folders_to_process=[new_folder_stripped],
            db_path_str=current_db_path,
            collection_obj=active_chroma_client_state_val.get_collection("images"),
            model_obj=active_model_state_val,
            processor_obj=active_processor_state_val,
            device_str=device,
            model_type_str=active_model_type_state_val,
            batch_size=batch_size_ui,
            progress_callback=update_gradio_progress,
        )

    except Exception as e:
        print(f"Error adding folder: {e}")
        gr.Error(f"Failed to add folder '{os.path.basename(new_folder_stripped)}': {e}")
        progress(1, desc="Error adding folder.")

    current_indexed_folders = read_indexed_folders(current_db_path)
    return "\n".join(current_indexed_folders)


def handle_delete_folder_button_click(
    folder_to_delete: str,
    current_db_path: str,
    active_chroma_client_state_val,
    progress=gr.Progress(track_tqdm=True),
):
    """Calls db_delete_folder directly to delete a folder from the active DB."""
    folder_to_delete_stripped = folder_to_delete.strip() if folder_to_delete else ""

    if not folder_to_delete_stripped:
        gr.Warning(
            "Folder path to delete cannot be empty. Please enter a path from the list above."
        )
        return "\n".join(read_indexed_folders(current_db_path))

    if not current_db_path:
        gr.Warning("No active model or DB path. Please select a model first.")
        return "\n".join(read_indexed_folders(current_db_path))

    if not active_chroma_client_state_val:
        gr.Warning("ChromaDB client not loaded. Please re-select a model.")
        return "\n".join(read_indexed_folders(current_db_path))

    progress(0, desc="Deleting folder from DB...")

    try:
        action_taken = db_delete_folder(
            folder_to_delete_str=folder_to_delete_stripped,
            db_path_str=current_db_path,
            collection_obj=active_chroma_client_state_val.get_collection("images"),
        )
        if action_taken:
            gr.Info(
                f"Folder deleted successfully: {os.path.basename(folder_to_delete_stripped)}"
            )
        else:
            gr.Warning(
                f"Folder not found: {os.path.basename(folder_to_delete_stripped)}"
            )
        progress(1, desc="Delete folder completed.")

    except Exception as e:
        print(f"Error deleting folder: {e}")
        gr.Error(
            f"Failed to delete folder '{os.path.basename(folder_to_delete_stripped)}': {e}"
        )
        progress(1, desc="Error.")

    current_indexed_folders = read_indexed_folders(current_db_path)
    return "\n".join(current_indexed_folders)


def update_duplicate_folder_dropdown(current_db_path: str):
    """Updates the duplicate folder dropdown with indexed folders."""
    if not current_db_path:
        return gr.Dropdown(choices=[], value=None)
    folders = read_indexed_folders(current_db_path)
    if not folders:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=folders, value=folders[0] if folders else None)


def get_relative_caption(abs_path: str, indexed_folders_text: str) -> str:
    """Returns the path relative to the indexed parent folder."""
    if not abs_path:
        return ""
    if not indexed_folders_text:
        return abs_path

    folders = [f.strip() for f in indexed_folders_text.split("\n") if f.strip()]
    folders.sort(key=len, reverse=True)

    abs_path_norm = os.path.normpath(os.path.abspath(abs_path))

    for folder in folders:
        folder_norm = os.path.normpath(os.path.abspath(folder))
        if abs_path_norm.startswith(folder_norm):
            try:
                return os.path.relpath(abs_path_norm, folder_norm)
            except ValueError:
                continue
    return abs_path


def handle_find_duplicates(
    duplicate_folder_path: str,
    duplicate_threshold: float,
    current_model_path: str,
    current_db_path: str,
    active_model_state_val,
    active_processor_state_val,
    active_model_type_state_val,
    active_chroma_client_state_val,
    batch_size_ui: int,
    verbose_ui: bool,
    initial_n_results_ui: int,
    indexed_folders_text: str,
    progress=gr.Progress(track_tqdm=True),
):
    """Finds duplicate images in a folder using the active model and ChromaDB."""
    if not duplicate_folder_path:
        gr.Warning("Please select a folder from the indexed folders.")
        return []

    if not current_model_path or not current_db_path:
        gr.Warning("No active model or DB path. Please select a model first.")
        return []

    if (
        not active_model_state_val
        or not active_processor_state_val
        or not active_model_type_state_val
        or not active_chroma_client_state_val
    ):
        gr.Warning(
            "Model components (model, processor, type, or DB client) not loaded. Please re-select a model."
        )
        return []

    if not os.path.isdir(duplicate_folder_path):
        print(f"Folder not found on disk: {os.path.abspath(duplicate_folder_path)}")
        gr.Warning(f"Folder not found: {duplicate_folder_path}")
        return []

    start_time = time.time()
    print(f"Processing duplicate search: {duplicate_folder_path}")
    print("Search Mode: DUPLICATES")

    try:
        # Find duplicate pairs
        pairs = find_duplicates_in_folder(
            folder_path=duplicate_folder_path,
            threshold=duplicate_threshold,
            batch_size=batch_size_ui,
            block_size=1024,
            recursive=True,
            active_model=active_model_state_val,
            active_processor=active_processor_state_val,
            active_model_type=active_model_type_state_val,
            active_chroma_client=active_chroma_client_state_val,
            device=device,
        )

        # Convert pairs to gallery format
        gallery_images = []
        for _, path1, path2 in pairs:
            if os.path.exists(path1):
                caption1 = get_relative_caption(path1, indexed_folders_text)
                gallery_images.append((path1, caption1))
            else:
                print(f"Missing: {path1}")

            if os.path.exists(path2):
                caption2 = get_relative_caption(path2, indexed_folders_text)
                gallery_images.append((path2, caption2))
            else:
                print(f"Missing: {path2}")

        total_images_found = len(gallery_images)
        gallery_images = gallery_images[:initial_n_results_ui]
        images_displayed = len(gallery_images)

        if not pairs:
            gr.Info("No duplicate or near-duplicate pairs found.")
        else:
            if total_images_found > initial_n_results_ui:
                gr.Info(
                    f"Found {len(pairs)} duplicate pairs ({total_images_found} images). "
                    f"Showing {images_displayed} images."
                )
            else:
                gr.Info(
                    f"Found {len(pairs)} duplicate pairs. Showing {images_displayed} images in gallery."
                )
            if verbose_ui:
                print(f"Found {len(pairs)} pairs:")
                for sim, p1, p2 in pairs:
                    print(f"{sim:.6f}\t{p1}\t{p2}")

        print(f"Returning {len(pairs)} duplicate pairs.")
        end_time = time.time()
        print(f"Search completed in {(end_time - start_time):.2f}s")
        return gallery_images

    except Exception as e:
        print(f"Error finding duplicates: {e}")
        gr.Error(f"Failed to find duplicates: {e}")
        progress(1, desc="Error.")
        end_time = time.time()
        print(f"Search completed in {(end_time - start_time):.2f}s")
        return []


def search(
    query: str,
    query_image_pil,
    initial_n_results_ui: int,
    logit_thresh_ui: float,
    clip_logit_thresh_ui: float,
    image_only_cosine_thresh_ui: float,
    combined_cosine_thresh_ui: float,
    current_model,
    current_processor,
    current_model_type: str,
    current_logit_scale_exp_val: float,
    current_logit_bias_val: float,
    current_chroma_client,
    verbose_ui: bool,
    indexed_folders_text: str,
):
    start_time = time.time()
    text_query_present = bool(query and query.strip())
    image_query_present = query_image_pil is not None
    text_emb_normalized_float32 = None
    image_emb_normalized_float32 = None
    gallery_images = []

    if not current_chroma_client or not current_model:
        gr.Warning("Model or Database not loaded. Please select a model.")
        return []
    if not text_query_present and not image_query_present:
        gr.Info("Please provide a text query or an image to search.")
        return []

    try:
        collection = current_chroma_client.get_collection("images")
    except Exception as e:
        print(f"Error getting collection: {e}")
        gr.Info(
            "Database collection 'images' not found or DB not initialized. "
            "Ensure `build-db.py` has been run for the selected model and then click 'Update/Sync Active DB'."
        )
        return []

    # Process Text Query (if provided)
    if text_query_present:
        print(f"Processing text query: '{query}'")
        with torch.no_grad():
            text_processing_args = {
                "text": query,
                "return_tensors": "pt",
                "truncation": True,
            }
            if current_model_type == "siglip":
                text_processing_args["padding"] = "max_length"
                text_processing_args["max_length"] = 64
            else:  # For CLIP
                text_processing_args["padding"] = True

            if current_model_type != "siglip" and "max_length" in text_processing_args:
                del text_processing_args["max_length"]

            inputs = current_processor(**text_processing_args).to(device)
            text_features_from_model = current_model.get_text_features(**inputs)
            text_features_float32 = text_features_from_model.to(torch.float32)

            if current_model_type == "siglip" or current_model_type == "clip":
                text_emb_normalized_float32 = F.normalize(
                    text_features_float32, p=2, dim=-1
                )
            else:
                text_emb_normalized_float32 = text_features_float32

    # Process Image Query (if provided)
    if image_query_present:
        print("Processing image query...")
        with torch.no_grad():
            inputs = current_processor(
                images=[query_image_pil], return_tensors="pt"
            ).to(device)
            query_image_features = current_model.get_image_features(**inputs)
            query_image_features_float32 = query_image_features.to(torch.float32)

            if current_model_type == "siglip" or current_model_type == "clip":
                image_emb_normalized_float32 = F.normalize(
                    query_image_features_float32, p=2, dim=-1
                )
            else:
                image_emb_normalized_float32 = query_image_features_float32

    # --- Search Mode Logic ---
    if text_query_present and not image_query_present:
        print("Search Mode: TEXT")
        try:
            query_results = collection.query(
                query_embeddings=text_emb_normalized_float32.cpu().squeeze(0).numpy(),
                n_results=int(initial_n_results_ui),
                include=[
                    "documents",
                    "embeddings",
                    "distances",
                ],  # Distances for CLIP, embeddings for SigLIP logit calc
            )
        except Exception as e:
            print(f"ChromaDB query error (Text): {e}")
            gr.Warning(f"Error searching images: {e}")
            return []

        if (
            query_results["documents"]
            and query_results["embeddings"]
            and query_results["documents"][0]
        ):
            candidates_count = len(query_results["documents"][0])
            print(f"Processing {candidates_count} candidates from DB (Text)")

            img_embeddings_list = query_results["embeddings"][0]
            all_img_embeddings_tensor = torch.tensor(
                img_embeddings_list, dtype=torch.float32
            ).to(device)

            # For text, cosine similarity is between text query embedding and DB image embeddings
            batch_cos_sim = F.cosine_similarity(
                text_emb_normalized_float32.squeeze(0),
                all_img_embeddings_tensor,
                dim=-1,
            )

            batch_logit_val_for_print = torch.zeros_like(batch_cos_sim)
            passes_threshold_mask = torch.zeros_like(batch_cos_sim, dtype=torch.bool)

            if current_model_type == "siglip":
                if (
                    current_logit_scale_exp_val is not None
                    and current_logit_bias_val is not None
                ):
                    batch_logit_val_for_print = (
                        batch_cos_sim * current_logit_scale_exp_val
                    ) + current_logit_bias_val
                    passes_threshold_mask = batch_logit_val_for_print > logit_thresh_ui
            elif current_model_type == "clip":
                if current_logit_scale_exp_val is not None:
                    batch_logit_val_for_print = (
                        batch_cos_sim * current_logit_scale_exp_val
                    )
                    passes_threshold_mask = (
                        batch_logit_val_for_print > clip_logit_thresh_ui
                    )

            doc_paths_all = query_results["documents"][0]
            passing_indices = torch.where(passes_threshold_mask)[0]

            for idx_tensor in passing_indices:
                idx = idx_tensor.item()
                doc_path = doc_paths_all[idx]
                if os.path.exists(doc_path):
                    caption = get_relative_caption(doc_path, indexed_folders_text)
                    gallery_images.append((doc_path, caption))
                else:
                    print(f"Missing: {doc_path}")

            if verbose_ui:
                for i in range(candidates_count):
                    doc_path = doc_paths_all[i]
                    cos_sim_val = batch_cos_sim[i].item()
                    logit_val = batch_logit_val_for_print[i].item()
                    sigmoid_val = torch.sigmoid(
                        torch.tensor(logit_val, device=device)
                    ).item()
                    status_char = "✓" if passes_threshold_mask[i].item() else "✗"
                    log_details = f"(cos: {cos_sim_val:.4f}, logit: {logit_val:.4f}, sigmoid: {sigmoid_val:.4f})"
                    print(f"{status_char} {doc_path} {log_details}")

    elif image_query_present and not text_query_present:
        print("Search Mode: IMAGE")
        try:
            query_results = collection.query(
                query_embeddings=image_emb_normalized_float32.cpu().squeeze(0).numpy(),
                n_results=int(initial_n_results_ui),
                include=["documents", "embeddings"],
            )
        except Exception as e:
            print(f"ChromaDB query error (Image): {e}")
            gr.Warning(f"Error searching images: {e}")
            return []

        if (
            query_results["documents"]
            and query_results["embeddings"]
            and query_results["documents"][0]
        ):
            doc_paths_all = query_results["documents"][0]
            db_img_embeddings_list = query_results["embeddings"][0]
            all_db_img_embeddings_tensor = torch.tensor(
                db_img_embeddings_list, dtype=torch.float32
            ).to(device)
            candidates_count = len(doc_paths_all)
            print(f"Processing {candidates_count} candidates from DB (Image)")

            batch_cos_sim_to_query_image = F.cosine_similarity(
                image_emb_normalized_float32.squeeze(0),
                all_db_img_embeddings_tensor,
                dim=-1,
            )
            passes_threshold_mask = (
                batch_cos_sim_to_query_image >= image_only_cosine_thresh_ui
            )

            passing_indices = torch.where(passes_threshold_mask)[0]
            temp_results = []  # To sort before adding to gallery_images
            for idx_tensor in passing_indices:
                idx = idx_tensor.item()
                temp_results.append(
                    (doc_paths_all[idx], batch_cos_sim_to_query_image[idx].item())
                )

            temp_results.sort(key=lambda x: x[1], reverse=True)

            for doc_path, cos_sim_val in temp_results:
                if os.path.exists(doc_path):
                    caption = get_relative_caption(doc_path, indexed_folders_text)
                    gallery_images.append((doc_path, caption))
                    if verbose_ui:
                        print(f"✓ {doc_path} (img_cos_sim: {cos_sim_val:.4f})")
                else:
                    print(f"Missing: {doc_path}")

            if verbose_ui and not gallery_images and doc_paths_all:
                print("No images passed the image cosine similarity threshold.")
                for i in range(len(doc_paths_all)):
                    print(
                        f"✗ {doc_paths_all[i]} (img_cos_sim: {batch_cos_sim_to_query_image[i].item():.4f})"
                    )

    elif text_query_present and image_query_present:
        print("Search Mode: COMBINED")
        # 1. Text Search Leg
        try:
            text_query_results = collection.query(
                query_embeddings=text_emb_normalized_float32.cpu().squeeze(0).numpy(),
                n_results=int(initial_n_results_ui),
                include=["documents", "embeddings"],
            )
        except Exception as e:
            print(f"ChromaDB query error (Combined - Text Leg): {e}")
            gr.Warning(f"Error in combined search (text leg): {e}")
            return []

        # 2. Image Search Leg
        try:
            image_query_results = collection.query(
                query_embeddings=image_emb_normalized_float32.cpu().squeeze(0).numpy(),
                n_results=int(initial_n_results_ui),
                include=["documents", "embeddings"],
            )
        except Exception as e:
            print(f"ChromaDB query error (Combined - Image Leg): {e}")
            gr.Warning(f"Error in combined search (image leg): {e}")
            return []

        candidate_data = {}  # {doc_path: {'db_embedding': tensor}}
        if text_query_results["documents"] and text_query_results["documents"][0]:
            for i, doc_path in enumerate(text_query_results["documents"][0]):
                if doc_path not in candidate_data:
                    candidate_data[doc_path] = {
                        "db_embedding": torch.tensor(
                            text_query_results["embeddings"][0][i], dtype=torch.float32
                        ).to(device)
                    }

        if image_query_results["documents"] and image_query_results["documents"][0]:
            for i, doc_path in enumerate(image_query_results["documents"][0]):
                if doc_path not in candidate_data:
                    candidate_data[doc_path] = {
                        "db_embedding": torch.tensor(
                            image_query_results["embeddings"][0][i], dtype=torch.float32
                        ).to(device)
                    }

        print(
            f"Processing {len(candidate_data)} unique candidates from combined search."
        )
        final_combined_results_data = (
            []
        )  # List of (doc_path, combined_score, text_sim, image_sim)

        candidate_paths = list(candidate_data.keys())
        if candidate_paths:
            # Batch process all candidates
            all_db_embeddings_tensor = torch.stack(
                [candidate_data[p]["db_embedding"] for p in candidate_paths]
            )

            # Batched cosine similarity
            text_sims_batch = F.cosine_similarity(
                text_emb_normalized_float32, all_db_embeddings_tensor
            )
            image_sims_batch = F.cosine_similarity(
                image_emb_normalized_float32, all_db_embeddings_tensor
            )

            # Calculate combined score and filter
            combined_scores_batch = (text_sims_batch + image_sims_batch) / 2.0
            passing_mask = combined_scores_batch >= combined_cosine_thresh_ui
            passing_indices = torch.where(passing_mask)[0]

            for idx in passing_indices:
                doc_path = candidate_paths[idx]
                score = combined_scores_batch[idx].item()
                ts = text_sims_batch[idx].item()
                Is = image_sims_batch[idx].item()
                final_combined_results_data.append((doc_path, score, ts, Is))

        final_combined_results_data.sort(key=lambda x: x[1], reverse=True)

        for doc_path, score, ts, Is in final_combined_results_data[
            : int(initial_n_results_ui)
        ]:
            if os.path.exists(doc_path):
                caption = get_relative_caption(doc_path, indexed_folders_text)
                gallery_images.append((doc_path, caption))
                if verbose_ui:
                    print(
                        f"✓ {doc_path} (comb_score: {score:.4f}, txt_sim: {ts:.4f}, img_sim: {Is:.4f})"
                    )
            else:
                print(f"Missing: {doc_path}")

        if verbose_ui and not gallery_images and candidate_data:
            print("No candidates passed the combined similarity threshold.")
            for doc_path, data_dict in candidate_data.items():
                db_img_embedding_tensor = data_dict["db_embedding"]
                text_sim = F.cosine_similarity(
                    text_emb_normalized_float32.squeeze(0),
                    db_img_embedding_tensor.unsqueeze(0),
                ).item()
                image_sim = F.cosine_similarity(
                    image_emb_normalized_float32.squeeze(0),
                    db_img_embedding_tensor.unsqueeze(0),
                ).item()
                combined_score = (text_sim + image_sim) / 2.0
                print(
                    f"✗ {doc_path} (comb: {combined_score:.4f}, txt: {text_sim:.4f}, img: {image_sim:.4f})"
                )

    result_count = len(gallery_images)
    if result_count == 0:
        print("No images found matching the criteria.")
    else:
        print(f"Returning {result_count} images.")

    end_time = time.time()
    print(f"Search completed in {(end_time - start_time):.2f}s")
    return gallery_images


def random_search(
    initial_n_results_ui: int,
    current_chroma_client,
    indexed_folders_text: str,
):
    """Returns a random selection of images from ChromaDB."""
    start_time = time.time()
    gallery_images = []

    if not current_chroma_client:
        gr.Warning("Database not loaded. Please select a model.")
        return []

    try:
        collection = current_chroma_client.get_collection("images")
    except Exception as e:
        print(f"Error getting collection: {e}")
        gr.Info(
            "Database collection 'images' not found or DB not initialized. "
            "Ensure `build-db.py` has been run for the selected model and then click 'Update/Sync Active DB'."
        )
        return []

    try:
        # Get all documents from ChromaDB
        all_results = collection.get(include=["documents"])

        if (
            not all_results
            or not all_results.get("documents")
            or not all_results["documents"]
        ):
            gr.Info("No images found in the database.")
            return []

        all_documents = all_results["documents"]
        total_count = len(all_documents)

        if total_count == 0:
            gr.Info("No images found in the database.")
            return []

        # Randomly sample the requested number of results
        n_results = min(int(initial_n_results_ui), total_count)
        sampled_indices = random.sample(range(total_count), n_results)

        print(
            f"Random search: Sampling {n_results} images from {total_count} total images."
        )

        for idx in sampled_indices:
            doc_path = all_documents[idx]
            if os.path.exists(doc_path):
                caption = get_relative_caption(doc_path, indexed_folders_text)
                gallery_images.append((doc_path, caption))
            else:
                print(f"Missing: {doc_path}")

        result_count = len(gallery_images)
        if result_count == 0:
            print("No valid images found after random sampling.")
        else:
            print(f"Returning {result_count} random images.")

        end_time = time.time()
        print(f"Random search completed in {(end_time - start_time):.2f}s")
        return gallery_images

    except Exception as e:
        print(f"Error in random search: {e}")
        gr.Error(f"Failed to retrieve random images: {e}")
        return []


def clear_search_and_gallery():
    """Clears the search query textbox, image input, and the results gallery."""
    return "", None, []


css_gallary = """
#gallery .grid-wrap {
    max-height: 900px !important;
}

/* Hide the caption on the thumbnail grid */
#gallery .caption-label {
    display: none !important;
}
"""

js_credits = """
function() {
    const footer = document.querySelector('footer');
    if (footer) {
        // Check if credits already exist
        if (footer.parentNode.querySelector('.locallens-credits')) {
            return;
        }
        const newContent = document.createElement('div');
        newContent.className = 'locallens-credits'; // Add a class for identification
        newContent.innerHTML = 'Made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

        newContent.style.textAlign = 'center';
        newContent.style.paddingTop = '50px';
        newContent.style.color = 'lightgray';

        // Style the hyperlink
        const link = newContent.querySelector('a');
        if (link) {
            link.style.color = 'gray';
            link.style.textDecoration = 'underline';
        }

        footer.parentNode.insertBefore(newContent, footer);
    }
}
"""

if __name__ == "__main__":
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="purple"),
        js=js_credits,
        css=css_gallary,
        title="Local Lens",
    ) as app:

        gr.Markdown("# Local Lens")

        active_model_path_state = gr.State(DEFAULT_MODEL_PATH)
        active_db_path_state = gr.State(None)

        model_state = gr.State(None)
        processor_state = gr.State(None)
        model_type_state = gr.State(None)
        logit_scale_state = gr.State(None)
        logit_bias_state = gr.State(None)
        chroma_client_state = gr.State(None)

        with gr.Row():
            # Left Column: Search, Model/DB Management, Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Search Query")
                query_textbox = gr.Textbox(
                    label="Query",
                    placeholder="Search here...",
                    show_label=False,
                )
                query_image_input = gr.Image(
                    type="pil",
                    label="Image Search (Optional)",
                    show_label=True,
                    elem_id="query_image",
                )
                with gr.Row():
                    search_button = gr.Button("Search", variant="primary", min_width=50)
                    feeling_lucky_button = gr.Button(
                        "I'm Feeling Lucky", variant="primary", min_width=50
                    )
                with gr.Row():
                    clear_button = gr.Button("Clear", min_width=50)

                gr.Markdown("### Model & Database")
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=None,
                    label="Select Model",
                    allow_custom_value=True,
                )
                with gr.Accordion("Database Management", open=False):
                    indexed_folders_display = gr.Textbox(
                        label="Indexed Folders", interactive=False, lines=3, max_lines=5
                    )
                    update_db_button = gr.Button(
                        "Full Update/Sync Active DB", variant="primary"
                    )

                    new_folder_textbox = gr.Textbox(
                        label="Folder Path to Add/Delete",
                        placeholder="/path/to/your/images",
                    )
                    with gr.Row():
                        add_folder_button = gr.Button("Add Folder", min_width=50)
                        delete_folder_button = gr.Button(
                            "Delete Folder", variant="stop", min_width=50
                        )

                gr.Markdown("### Find Duplicates")
                with gr.Accordion("Find Duplicates", open=False):
                    duplicate_folder_dropdown = gr.Dropdown(
                        label="Select Indexed Folder to Scan",
                        choices=[],
                        value=None,
                        info="Select a folder from the indexed folders above.",
                    )
                    duplicate_threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.97,
                        label="Similarity Threshold",
                        info="Cosine similarity threshold for reporting duplicates. Higher = more strict.",
                    )
                    find_duplicates_button = gr.Button(
                        "Find Duplicates", variant="primary"
                    )

                gr.Markdown("### Advanced Parameters")
                with gr.Accordion("Advanced Parameters", open=False):
                    initial_n_results_slider = gr.Slider(
                        minimum=1,
                        maximum=500,
                        step=1,
                        value=INITIAL_N_RESULTS,
                        label="Initial N Results (Max Images)",
                        info="Images to fetch from DB before applying confidence thresholds.",
                    )
                    siglip_thresh_slider = gr.Slider(
                        minimum=-15.0,
                        maximum=15.0,
                        step=0.1,
                        value=FALLBACK_SIGLIP_THRESHOLD,
                        label="Logit Confidence Threshold (SigLIP)",
                        info="SigLIP model confidence. Higher values = more confident.",
                    )
                    clip_thresh_slider = gr.Slider(
                        minimum=0.0,
                        maximum=40.0,
                        step=0.1,
                        value=FALLBACK_CLIP_THRESHOLD,
                        label="Logit Confidence Threshold (CLIP)",
                        info="CLIP model confidence. Higher values = more confident.",
                    )
                    image_only_cosine_thresh_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=FALLBACK_IMAGE_ONLY_THRESHOLD,
                        label="Cosine Similarity Threshold (Image Only)",
                        info="For reverse image search (image input only). Higher values = more confident.",
                    )
                    combined_cosine_thresh_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=FALLBACK_COMBINED_THRESHOLD,
                        label="Cosine Similarity Threshold (Combined)",
                        info="For text + image search. Higher values = more confident.",
                    )
                    batch_size_slider = gr.Slider(
                        minimum=4,
                        maximum=256,
                        step=4,
                        value=64,
                        label="Processing Batch Size",
                        info="Images to process in one batch during add/update. Higher = more memory usage.",
                    )
                    verbose_checkbox = gr.Checkbox(
                        value=False,
                        label="Verbose Logging",
                        info="Show detailed confidence scores and similarity metrics in console output.",
                    )

            # Right Column: Gallery
            with gr.Column(scale=3):
                results_gallery = gr.Gallery(
                    label="Search Results",
                    columns=3,
                    show_label=True,
                    elem_id="gallery",
                )

        load_outputs = [
            active_model_path_state,
            active_db_path_state,
            model_state,
            processor_state,
            model_type_state,
            logit_scale_state,
            logit_bias_state,
            chroma_client_state,
            indexed_folders_display,
            siglip_thresh_slider,
            clip_thresh_slider,
            combined_cosine_thresh_slider,
            image_only_cosine_thresh_slider,
            model_dropdown,
        ]

        def update_dropdown_from_indexed_folders(indexed_folders_text: str):
            """Update duplicate dropdown from indexed folders text."""
            if not indexed_folders_text:
                return gr.Dropdown(choices=[], value=None)
            folders = [f.strip() for f in indexed_folders_text.split("\n") if f.strip()]
            if not folders:
                return gr.Dropdown(choices=[], value=None)
            return gr.Dropdown(choices=folders, value=folders[0] if folders else None)

        app.load(
            fn=load_and_switch_model_db,
            inputs=[active_model_path_state],
            outputs=load_outputs,
            show_progress="full",
        ).then(
            fn=update_dropdown_from_indexed_folders,
            inputs=[indexed_folders_display],
            outputs=[duplicate_folder_dropdown],
        )

        model_dropdown.change(
            fn=load_and_switch_model_db,
            inputs=[model_dropdown],
            outputs=load_outputs,
            show_progress="full",
        ).then(
            fn=update_dropdown_from_indexed_folders,
            inputs=[indexed_folders_display],
            outputs=[duplicate_folder_dropdown],
        )

        update_db_button.click(
            fn=handle_update_sync_button_click,
            inputs=[
                active_model_path_state,
                active_db_path_state,
                model_state,
                processor_state,
                model_type_state,
                chroma_client_state,
                batch_size_slider,
            ],
            outputs=[],
            show_progress="full",
        )

        def handle_add_folder_with_dropdown_update(*args):
            """Wrapper to update both indexed folders display and duplicate dropdown."""
            indexed_folders_result = handle_add_folder_button_click(*args)
            dropdown_update = update_dropdown_from_indexed_folders(
                indexed_folders_result
            )
            return indexed_folders_result, dropdown_update

        def handle_delete_folder_with_dropdown_update(*args):
            """Wrapper to update both indexed folders display and duplicate dropdown."""
            indexed_folders_result = handle_delete_folder_button_click(*args)
            dropdown_update = update_dropdown_from_indexed_folders(
                indexed_folders_result
            )
            return indexed_folders_result, dropdown_update

        add_folder_button.click(
            fn=handle_add_folder_with_dropdown_update,
            inputs=[
                new_folder_textbox,
                active_model_path_state,
                active_db_path_state,
                model_state,
                processor_state,
                model_type_state,
                chroma_client_state,
                batch_size_slider,
            ],
            outputs=[indexed_folders_display, duplicate_folder_dropdown],
            show_progress="full",
        )

        delete_folder_button.click(
            fn=handle_delete_folder_with_dropdown_update,
            inputs=[new_folder_textbox, active_db_path_state, chroma_client_state],
            outputs=[indexed_folders_display, duplicate_folder_dropdown],
            show_progress="full",
        )

        find_duplicates_button.click(
            fn=handle_find_duplicates,
            inputs=[
                duplicate_folder_dropdown,
                duplicate_threshold_slider,
                active_model_path_state,
                active_db_path_state,
                model_state,
                processor_state,
                model_type_state,
                chroma_client_state,
                batch_size_slider,
                verbose_checkbox,
                initial_n_results_slider,
                indexed_folders_display,
            ],
            outputs=[results_gallery],
            show_progress="full",
        )

        search_inputs = [
            query_textbox,
            query_image_input,
            initial_n_results_slider,
            siglip_thresh_slider,
            clip_thresh_slider,
            image_only_cosine_thresh_slider,
            combined_cosine_thresh_slider,
            model_state,
            processor_state,
            model_type_state,
            logit_scale_state,
            logit_bias_state,
            chroma_client_state,
            verbose_checkbox,
            indexed_folders_display,
        ]

        query_textbox.submit(fn=search, inputs=search_inputs, outputs=[results_gallery])
        search_button.click(fn=search, inputs=search_inputs, outputs=[results_gallery])

        random_search_inputs = [
            initial_n_results_slider,
            chroma_client_state,
            indexed_folders_display,
        ]
        feeling_lucky_button.click(
            fn=random_search, inputs=random_search_inputs, outputs=[results_gallery]
        )

        clear_button.click(
            fn=clear_search_and_gallery,
            inputs=[],
            outputs=[query_textbox, query_image_input, results_gallery],
        )

    # Gradio requires explicit permission to pass file paths to the gallery.
    # Linux/Mac: allow root ("/"), Windows: add drive letters A-Z.
    allowed_paths = ["/"]
    if os.name == "nt":
        allowed_paths = [f"{chr(d)}:\\" for d in range(ord("A"), ord("Z") + 1)]

    app.launch(inbrowser=True, allowed_paths=allowed_paths)

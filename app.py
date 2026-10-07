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

import functools
import html
import os
import time
from pathlib import Path

import chromadb
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from build_db import db_add_folders, db_delete_folder, db_update_indexed_folders
from device import get_best_device, get_best_dtype
from find_duplicates import (
    filter_gallery_items,
    filter_resolved_pairs,
    find_duplicates_in_folder,
    find_duplicates_in_folders,
)
from index_store import (
    IndexStore,
    get_media_file_info,
    is_path_under_folder,
    migrate_legacy_db,
    path_and_caption_from_result,
    query_filter_for_folder,
    relative_caption,
    reveal_in_file_manager,
    trash_and_delete_indexed_media,
)
from model_utils import load_model_and_processor

__version__ = "1.6.2"

# --- Configuration ---
AVAILABLE_MODELS = [
    "google/siglip2-so400m-patch14-384",
    "google/siglip2-so400m-patch16-512",
    "apple/DFN5B-CLIP-ViT-H-14-378",
    "google/siglip2-giant-opt-patch16-384",
    "facebook/metaclip-h14-fullcc2.5b",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "openai/clip-vit-large-patch14",
    "google/siglip2-base-patch16-224",
]
DEFAULT_MODEL_PATH = None

INITIAL_N_RESULTS = 50  # Number of results to fetch from ChromaDB before filtering

FALLBACK_SIGLIP_THRESHOLD = -8.0
FALLBACK_CLIP_THRESHOLD = 20.0
FALLBACK_COMBINED_THRESHOLD = 0.45
FALLBACK_IMAGE_ONLY_THRESHOLD = 0.70
MODEL_CONFIDENCE_DEFAULTS = {
    "google/siglip2-so400m-patch14-384": {
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
    "google/siglip2-giant-opt-patch16-384": {
        "siglip_thresh": -8.0,
        "combined_thresh": 0.50,
        "image_only_thresh": 0.85,
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
    "google/siglip2-base-patch16-224": {
        "siglip_thresh": -8.0,
        "combined_thresh": 0.50,
        "image_only_thresh": 0.85,
    },
}
FALLBACK_BATCH_SIZE = 64
MODEL_BATCH_SIZE_DEFAULTS = {
    "google/siglip2-so400m-patch14-384": 64,
    "google/siglip2-so400m-patch16-512": 64,
    "apple/DFN5B-CLIP-ViT-H-14-378": 32,
    "google/siglip2-giant-opt-patch16-384": 16,
    "facebook/metaclip-h14-fullcc2.5b": 64,
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 64,
    "openai/clip-vit-large-patch14": 64,
    "google/siglip2-base-patch16-224": 128,
}

device = get_best_device()
dtype = get_best_dtype(device)


def get_simplified_model_identifier(model_path_str: str) -> str:
    """Extracts a simplified model identifier (type+size) from the model path."""
    name_lower = model_path_str.lower()
    parts = name_lower.split("/")
    model_name_part = parts[-1]

    if "siglip2" in model_name_part:
        if "so400m" in model_name_part and "384" in model_name_part:
            return "siglip2-so400m-384"
        sub_parts = model_name_part.split("-")
        if len(sub_parts) >= 2:
            return f"{sub_parts[0]}-{sub_parts[1]}"  # "siglip2-so400m", "siglip2-base", "siglip2-giant"
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
    """Reads indexed folders from the SQLite registry, falling back to the legacy text file."""
    if not db_path:
        return []
    try:
        folders = IndexStore(db_path).list_folders()
        if folders:
            return folders
    except Exception as e:
        print(f"Error reading indexed folders registry: {e}")

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
        "",
        None,
        None,
        None,
        None,
        gr.Dropdown(interactive=False),
        None,
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
            "",
            gr.Slider(value=FALLBACK_SIGLIP_THRESHOLD, visible=False),
            gr.Slider(value=FALLBACK_CLIP_THRESHOLD, visible=False),
            FALLBACK_COMBINED_THRESHOLD,
            FALLBACK_IMAGE_ONLY_THRESHOLD,
            gr.Dropdown(interactive=True),
            FALLBACK_BATCH_SIZE,
        )
        return

    progress(0, desc="Starting model...")
    ui_siglip_thresh_to_set = FALLBACK_SIGLIP_THRESHOLD
    ui_clip_thresh_to_set = FALLBACK_CLIP_THRESHOLD
    ui_combined_thresh_to_set = FALLBACK_COMBINED_THRESHOLD
    ui_image_only_thresh_to_set = FALLBACK_IMAGE_ONLY_THRESHOLD
    ui_batch_size_to_set = MODEL_BATCH_SIZE_DEFAULTS.get(
        selected_model_path, FALLBACK_BATCH_SIZE
    )
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
            collection = new_chroma_client.get_or_create_collection(
                name="images", metadata={"hnsw:space": "cosine"}
            )
            migrated_count = migrate_legacy_db(new_db_path, collection)
            if migrated_count:
                progress(0.85, desc=f"Migrated {migrated_count} legacy records.")
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
            gr.Slider(
                value=ui_siglip_thresh_to_set, visible=new_model_type == "siglip"
            ),
            gr.Slider(value=ui_clip_thresh_to_set, visible=new_model_type == "clip"),
            ui_combined_thresh_to_set,
            ui_image_only_thresh_to_set,
            gr.Dropdown(interactive=True),
            ui_batch_size_to_set,
        )
    except Exception as e:
        print(f"Error loading model/DB: {e}")
        gr.Warning(
            f"Error loading model/DB for {selected_model_path}: {e}", title="Error"
        )
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
            gr.Slider(value=FALLBACK_SIGLIP_THRESHOLD, visible=False),
            gr.Slider(value=FALLBACK_CLIP_THRESHOLD, visible=False),
            FALLBACK_COMBINED_THRESHOLD,
            FALLBACK_IMAGE_ONLY_THRESHOLD,
            gr.Dropdown(interactive=True),
            FALLBACK_BATCH_SIZE,
        )


def make_index_progress_callback(progress):
    """Maps build_db progress events onto a Gradio progress bar."""

    def callback(status, **kwargs):
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
            batch_duration = kwargs.get("batch_duration_seconds")

            prog_fraction = (
                cumulative_processed / total_images_to_process
                if total_images_to_process > 0
                else 0
            )
            duration_part = (
                f" in {batch_duration:.2f}s" if batch_duration is not None else ""
            )
            desc_str = (
                f"Batch {current_batch}/{total_batches} "
                f"({images_in_batch} images{duration_part}). "
                f"Total: {cumulative_processed}/{total_images_to_process}"
            )
            progress(prog_fraction, desc=desc_str)
        elif status == "all_batches_done":
            total_added = kwargs.get("total_successfully_added", 0)
            progress(1, desc=f"Successfully processed {total_added} images.")

    return callback


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
        added, removed = db_update_indexed_folders(
            db_path_str=current_db_path,
            collection_obj=active_chroma_client_state_val.get_collection("images"),
            model_obj=active_model_state_val,
            processor_obj=active_processor_state_val,
            device=device,
            batch_size=batch_size_ui,
            progress_callback=make_index_progress_callback(progress),
        )
        if added or removed:
            gr.Info(f"Sync complete: {added} added or updated, {removed} removed.")
        else:
            gr.Info("Sync complete: database is already up to date.")
        progress(1, desc="DB updated.")

    except Exception as e:
        print(f"Error updating DB directly: {e}")
        gr.Warning(
            f"An unexpected error occurred while trying to update DB '{current_db_path}': {e}",
            title="Error",
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

    index_progress = make_index_progress_callback(progress)

    def update_gradio_progress(status, **kwargs):
        if status == "add_folder_completed":
            duration = kwargs.get("duration_seconds", 0)
            gr.Info(
                f"Folder '{os.path.basename(new_folder_stripped)}' processed in {duration:.2f}s."
            )
            progress(1, desc=f"Add folder completed in {duration:.2f}s.")
        else:
            index_progress(status, **kwargs)

    try:
        db_add_folders(
            folders_to_process=[new_folder_stripped],
            db_path_str=current_db_path,
            collection_obj=active_chroma_client_state_val.get_collection("images"),
            model_obj=active_model_state_val,
            processor_obj=active_processor_state_val,
            device=device,
            batch_size=batch_size_ui,
            progress_callback=update_gradio_progress,
        )

    except Exception as e:
        print(f"Error adding folder: {e}")
        gr.Warning(
            f"Failed to add folder '{os.path.basename(new_folder_stripped)}': {e}",
            title="Error",
        )
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
        gr.Warning("No folder selected. Choose an indexed folder to delete.")
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
        gr.Warning(
            f"Failed to delete folder '{os.path.basename(folder_to_delete_stripped)}': {e}",
            title="Error",
        )
        progress(1, desc="Error.")

    current_indexed_folders = read_indexed_folders(current_db_path)
    return "\n".join(current_indexed_folders)


def render_index_summary(db_path: str | None):
    """Per-folder and total indexed item counts for the Database Management table."""
    if not db_path:
        return gr.Dataframe(value=[], label="Indexed Folders")
    counts, total = IndexStore(db_path).folder_item_counts()
    return gr.Dataframe(
        value=[[folder, f"{count:,}"] for folder, count in counts],
        label=f"Indexed Folders ({total:,} items total)",
    )


def parse_indexed_folders_text(indexed_folders_text: str) -> list[str]:
    return [
        f.strip()
        for f in (indexed_folders_text or "").split("\n")
        if f.strip() and os.path.sep in f.strip()
    ]


def update_duplicate_folder_dropdown(current_db_path: str):
    """Updates the duplicate folder dropdown with indexed folders."""
    if not current_db_path:
        return gr.Dropdown(choices=[], value=None)
    folders = read_indexed_folders(current_db_path)
    if not folders:
        return gr.Dropdown(choices=[], value=None)
    choices = ["All"] + folders
    return gr.Dropdown(choices=choices, value="All")


def update_active_folder_dropdown(indexed_folders_text: str):
    """Updates the active folder dropdown from indexed folders text, with 'All' as default."""
    if not indexed_folders_text:
        return gr.Dropdown(choices=[], value=None)
    folders = parse_indexed_folders_text(indexed_folders_text)
    if not folders:
        return gr.Dropdown(choices=[], value=None)
    choices = ["All"] + folders
    return gr.Dropdown(choices=choices, value="All")


def filter_gallery_by_active_folder(gallery_images: list, active_folder: str) -> list:
    """Filters gallery images to only include those under the active folder."""
    if not active_folder or active_folder == "All":
        return gallery_images
    return [
        (path, caption)
        for path, caption in gallery_images
        if is_path_under_folder(path, active_folder)
    ]


def get_relative_caption(abs_path: str, indexed_folders_text: str) -> str:
    """Returns the path relative to the indexed parent folder."""
    if not abs_path:
        return ""
    if not indexed_folders_text:
        return abs_path

    folders = [f.strip() for f in indexed_folders_text.split("\n") if f.strip()]
    folders.sort(key=len, reverse=True)

    for folder in folders:
        if is_path_under_folder(abs_path, folder):
            return relative_caption(abs_path, folder)
    return abs_path


def render_duplicate_pair_view(
    pairs: list,
    current_idx: int,
    indexed_folders_text: str,
):
    """Generates UI updates for the side-by-side duplicate reviewer."""
    if not pairs or current_idx < 0 or current_idx >= len(pairs):
        return (
            gr.Column(visible=False),
            0,
            "No duplicate pairs to review",
            None,
            "",
            None,
            "",
            gr.Button(value="Move to Recycle Bin", interactive=False),
            gr.Button(value="Move to Recycle Bin", interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
        )

    sim, path_a, path_b = pairs[current_idx]
    folders = (
        parse_indexed_folders_text(indexed_folders_text) if indexed_folders_text else []
    )
    info_a = get_media_file_info(path_a, base_folders=folders)
    info_b = get_media_file_info(path_b, base_folders=folders)

    status_text = (
        f"**Duplicate Pair {current_idx + 1} of {len(pairs)}** &nbsp;&bull;&nbsp; "
        f"**{sim * 100:.1f}%** Similarity"
    )

    def format_card_html(info: dict) -> str:
        if not info.get("exists"):
            missing_path = html.escape(str(info.get("path", "")))
            return (
                f"<div class='dup-info-card'>"
                f"  <div class='dup-info-caption' style='color: var(--body-text-color-subdued, #888);'>File Missing or Removed</div>"
                f"  <div class='dup-info-path' title='{missing_path}'>{missing_path}</div>"
                f"</div>"
            )

        caption = html.escape(str(info.get("caption", "")))
        full_path = html.escape(str(info.get("path", "")))
        dim_str = html.escape(str(info.get("dim_str", "Unknown")))
        size_str = html.escape(str(info.get("size_str", "0 B")))
        mtime_str = html.escape(str(info.get("mtime_str", "Unknown")))

        return (
            f"<div class='dup-info-card'>"
            f"  <div class='dup-info-caption' title='{full_path}'>{caption}</div>"
            f"  <div class='dup-info-grid'>"
            f"    <div class='dup-info-row'>"
            f"      <span class='dup-info-label'>Resolution</span>"
            f"      <span class='dup-info-val'>{dim_str}</span>"
            f"    </div>"
            f"    <div class='dup-info-row'>"
            f"      <span class='dup-info-label'>File Size</span>"
            f"      <span class='dup-info-val'>{size_str}</span>"
            f"    </div>"
            f"    <div class='dup-info-row'>"
            f"      <span class='dup-info-label'>Modified</span>"
            f"      <span class='dup-info-val'>{mtime_str}</span>"
            f"    </div>"
            f"    <div class='dup-info-row'>"
            f"      <span class='dup-info-label'>Location</span>"
            f"      <span class='dup-info-val dup-info-path' title='{full_path}'>{full_path}</span>"
            f"    </div>"
            f"  </div>"
            f"</div>"
        )

    info_a_html = format_card_html(info_a)
    info_b_html = format_card_html(info_b)

    can_trash_a = info_a["exists"]
    can_trash_b = info_b["exists"]

    label_a = "Move to Recycle Bin" if can_trash_a else "Remove Record"
    label_b = "Move to Recycle Bin" if can_trash_b else "Remove Record"

    has_prev = current_idx > 0
    has_next = current_idx < len(pairs) - 1

    return (
        gr.Column(visible=True),
        current_idx,
        status_text,
        path_a if can_trash_a else None,
        info_a_html,
        path_b if can_trash_b else None,
        info_b_html,
        gr.Button(value=label_a, interactive=True),
        gr.Button(value=label_b, interactive=True),
        gr.Button(interactive=has_prev),
        gr.Button(interactive=has_next),
    )


def handle_duplicate_nav(
    step: int,
    current_idx: int,
    pairs: list,
    indexed_folders_text: str,
):
    """Navigate backward or forward through duplicate pairs."""
    if not pairs:
        new_idx = 0
    else:
        new_idx = max(0, min(len(pairs) - 1, current_idx + step))
    return render_duplicate_pair_view(pairs, new_idx, indexed_folders_text)


def handle_trash_duplicate(
    target_side: str,
    current_idx: int,
    pairs: list,
    gallery_images: list,
    current_db_path: str,
    chroma_client,
    indexed_folders_text: str,
):
    """Trashes the selected image from a duplicate pair and updates the reviewer."""
    if not pairs or current_idx < 0 or current_idx >= len(pairs):
        return (
            gallery_images,
            pairs,
            gr.Tabs(selected="tab_search", elem_classes=["hide-tabs-nav"]),
            *render_duplicate_pair_view([], 0, indexed_folders_text),
        )

    sim, path_a, path_b = pairs[current_idx]
    folders = (
        parse_indexed_folders_text(indexed_folders_text) if indexed_folders_text else []
    )
    info_a = get_media_file_info(path_a, base_folders=folders)
    info_b = get_media_file_info(path_b, base_folders=folders)

    if target_side == "left":
        target_path = path_a
        target_info = info_a
    elif target_side == "right":
        target_path = path_b
        target_info = info_b
    else:
        return (
            gallery_images,
            pairs,
            gr.Tabs(selected="tab_duplicates", elem_classes=["show-tabs-nav"]),
            *render_duplicate_pair_view(pairs, current_idx, indexed_folders_text),
        )

    collection = None
    if chroma_client:
        try:
            collection = chroma_client.get_collection("images")
        except Exception as e:
            print(f"Error accessing collection: {e}")

    file_existed = target_info.get("exists", False)
    success = trash_and_delete_indexed_media(target_path, current_db_path, collection)
    if success:
        if not file_existed:
            file_name = os.path.basename(target_path)
            gr.Info(f"Removed missing record '{file_name}' from index.")
    else:
        gr.Warning(
            f"Could not trash '{target_path}'. File may be locked by another process or permission denied."
        )
        return (
            gallery_images,
            pairs,
            gr.Tabs(selected="tab_duplicates", elem_classes=["show-tabs-nav"]),
            *render_duplicate_pair_view(pairs, current_idx, indexed_folders_text),
        )

    remaining_pairs = filter_resolved_pairs(pairs, target_path)
    updated_gallery = filter_gallery_items(gallery_images, target_path)

    new_idx = (
        max(0, min(len(remaining_pairs) - 1, current_idx)) if remaining_pairs else 0
    )
    if not remaining_pairs:
        gr.Info("All duplicate pairs have been resolved.")
        selected_tabs = gr.Tabs(selected="tab_search", elem_classes=["hide-tabs-nav"])
    else:
        selected_tabs = gr.Tabs(
            selected="tab_duplicates", elem_classes=["show-tabs-nav"]
        )

    view_updates = render_duplicate_pair_view(
        remaining_pairs, new_idx, indexed_folders_text
    )
    return (
        updated_gallery,
        remaining_pairs,
        selected_tabs,
        *view_updates,
    )


def empty_duplicate_results():
    return (
        [],
        [],
        gr.Tabs(selected="tab_search", elem_classes=["hide-tabs-nav"]),
        *render_duplicate_pair_view([], 0, ""),
    )


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
        return empty_duplicate_results()

    if not current_model_path or not current_db_path:
        gr.Warning("No active model or DB path. Please select a model first.")
        return empty_duplicate_results()

    if (
        not active_model_state_val
        or not active_processor_state_val
        or not active_model_type_state_val
        or not active_chroma_client_state_val
    ):
        gr.Warning(
            "Model components (model, processor, type, or DB client) not loaded. Please re-select a model."
        )
        return empty_duplicate_results()

    scan_all_folders = duplicate_folder_path == "All"
    if scan_all_folders:
        duplicate_folders = parse_indexed_folders_text(indexed_folders_text)
        missing_folders = [
            folder for folder in duplicate_folders if not os.path.isdir(folder)
        ]
        duplicate_folders = [
            folder for folder in duplicate_folders if os.path.isdir(folder)
        ]
        if not duplicate_folders:
            gr.Warning("No indexed folders are available to scan.")
            return empty_duplicate_results()
        if missing_folders:
            print(
                "Skipping missing indexed folders: "
                + ", ".join(os.path.abspath(folder) for folder in missing_folders)
            )
            gr.Warning(f"Skipping {len(missing_folders)} missing indexed folder(s).")
    elif not os.path.isdir(duplicate_folder_path):
        print(f"Folder not found on disk: {os.path.abspath(duplicate_folder_path)}")
        gr.Warning(f"Folder not found: {duplicate_folder_path}")
        return empty_duplicate_results()
    else:
        duplicate_folders = [duplicate_folder_path]

    start_time = time.time()
    print(f"Processing duplicate search: {duplicate_folder_path}")
    print("Search Mode: DUPLICATES")

    try:
        if scan_all_folders:
            pairs = find_duplicates_in_folders(
                folder_paths=duplicate_folders,
                threshold=duplicate_threshold,
                batch_size=batch_size_ui,
                block_size=1024,
                recursive=True,
                active_model=active_model_state_val,
                active_processor=active_processor_state_val,
                active_chroma_client=active_chroma_client_state_val,
                db_path=current_db_path,
                device=device,
            )
        else:
            pairs = find_duplicates_in_folder(
                folder_path=duplicate_folder_path,
                threshold=duplicate_threshold,
                batch_size=batch_size_ui,
                block_size=1024,
                recursive=True,
                active_model=active_model_state_val,
                active_processor=active_processor_state_val,
                active_chroma_client=active_chroma_client_state_val,
                db_path=current_db_path,
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
        view_updates = render_duplicate_pair_view(pairs, 0, indexed_folders_text)
        if not pairs:
            selected_tabs = gr.Tabs(
                selected="tab_search", elem_classes=["hide-tabs-nav"]
            )
        else:
            selected_tabs = gr.Tabs(
                selected="tab_search", elem_classes=["show-tabs-nav"]
            )
        return (
            gallery_images,
            pairs,
            selected_tabs,
            *view_updates,
        )

    except Exception as e:
        print(f"Error finding duplicates: {e}")
        gr.Warning(f"Failed to find duplicates: {e}", title="Error")
        progress(1, desc="Error.")
        end_time = time.time()
        print(f"Search completed in {(end_time - start_time):.2f}s")
        return empty_duplicate_results()


def caption_with_score(caption: str, score: float, show: bool) -> str:
    """Appends the score a search mode thresholds on, so captions match the sliders."""
    return f"{caption} · {score:.3f}" if show else caption


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
    current_db_path: str,
    verbose_ui: bool,
    active_folder: str = "All",
    show_scores: bool = False,
):
    start_time = time.time()
    text_query_present = bool(query and query.strip())
    image_query_present = query_image_pil is not None
    text_emb_normalized_float32 = None
    image_emb_normalized_float32 = None
    gallery_images = []

    if not current_chroma_client or not current_model or not current_db_path:
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

    folder_filter = query_filter_for_folder(active_folder)

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
            query_kwargs = {
                "query_embeddings": text_emb_normalized_float32.cpu()
                .squeeze(0)
                .numpy(),
                "n_results": int(initial_n_results_ui),
                "include": [
                    "documents",
                    "embeddings",
                    "distances",
                    "metadatas",
                ],  # Distances for CLIP, embeddings for SigLIP logit calc
            }
            query_kwargs.update(folder_filter)

            query_results = collection.query(**query_kwargs)
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
                np.asarray(img_embeddings_list), dtype=torch.float32
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
            metadatas_all = (query_results.get("metadatas") or [[]])[0] or []
            passing_indices = torch.where(passes_threshold_mask)[0]

            for idx_tensor in passing_indices:
                idx = idx_tensor.item()
                metadata = metadatas_all[idx] if idx < len(metadatas_all) else None
                doc_path, caption = path_and_caption_from_result(
                    doc_paths_all[idx], metadata
                )
                if os.path.exists(doc_path):
                    score = batch_logit_val_for_print[idx].item()
                    gallery_images.append(
                        (doc_path, caption_with_score(caption, score, show_scores))
                    )
                else:
                    print(f"Missing: {doc_path}")

            if verbose_ui:
                for i in range(candidates_count):
                    metadata = metadatas_all[i] if i < len(metadatas_all) else None
                    doc_path, _ = path_and_caption_from_result(
                        doc_paths_all[i], metadata
                    )
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
            query_kwargs = {
                "query_embeddings": image_emb_normalized_float32.cpu()
                .squeeze(0)
                .numpy(),
                "n_results": int(initial_n_results_ui),
                "include": ["documents", "embeddings", "metadatas"],
            }
            query_kwargs.update(folder_filter)

            query_results = collection.query(**query_kwargs)
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
            metadatas_all = (query_results.get("metadatas") or [[]])[0] or []
            db_img_embeddings_list = query_results["embeddings"][0]
            all_db_img_embeddings_tensor = torch.tensor(
                np.asarray(db_img_embeddings_list), dtype=torch.float32
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
                metadata = metadatas_all[idx] if idx < len(metadatas_all) else None
                doc_path, caption = path_and_caption_from_result(
                    doc_paths_all[idx], metadata
                )
                temp_results.append(
                    (doc_path, caption, batch_cos_sim_to_query_image[idx].item())
                )

            temp_results.sort(key=lambda x: x[2], reverse=True)

            for doc_path, caption, cos_sim_val in temp_results:
                if os.path.exists(doc_path):
                    gallery_images.append(
                        (
                            doc_path,
                            caption_with_score(caption, cos_sim_val, show_scores),
                        )
                    )
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
            text_query_kwargs = {
                "query_embeddings": text_emb_normalized_float32.cpu()
                .squeeze(0)
                .numpy(),
                "n_results": int(initial_n_results_ui),
                "include": ["documents", "embeddings", "metadatas"],
            }
            text_query_kwargs.update(folder_filter)

            text_query_results = collection.query(**text_query_kwargs)
        except Exception as e:
            print(f"ChromaDB query error (Combined - Text Leg): {e}")
            gr.Warning(f"Error in combined search (text leg): {e}")
            return []

        # 2. Image Search Leg
        try:
            image_query_kwargs = {
                "query_embeddings": image_emb_normalized_float32.cpu()
                .squeeze(0)
                .numpy(),
                "n_results": int(initial_n_results_ui),
                "include": ["documents", "embeddings", "metadatas"],
            }
            image_query_kwargs.update(folder_filter)

            image_query_results = collection.query(**image_query_kwargs)
        except Exception as e:
            print(f"ChromaDB query error (Combined - Image Leg): {e}")
            gr.Warning(f"Error in combined search (image leg): {e}")
            return []

        candidate_data = {}
        if text_query_results["documents"] and text_query_results["documents"][0]:
            text_metadatas = (text_query_results.get("metadatas") or [[]])[0] or []
            for i, doc_path in enumerate(text_query_results["documents"][0]):
                metadata = text_metadatas[i] if i < len(text_metadatas) else None
                result_path, caption = path_and_caption_from_result(doc_path, metadata)
                media_id = (metadata or {}).get("media_id") or result_path
                if media_id not in candidate_data:
                    candidate_data[media_id] = {
                        "path": result_path,
                        "caption": caption,
                        "db_embedding": torch.tensor(
                            text_query_results["embeddings"][0][i], dtype=torch.float32
                        ).to(device),
                    }

        if image_query_results["documents"] and image_query_results["documents"][0]:
            image_metadatas = (image_query_results.get("metadatas") or [[]])[0] or []
            for i, doc_path in enumerate(image_query_results["documents"][0]):
                metadata = image_metadatas[i] if i < len(image_metadatas) else None
                result_path, caption = path_and_caption_from_result(doc_path, metadata)
                media_id = (metadata or {}).get("media_id") or result_path
                if media_id not in candidate_data:
                    candidate_data[media_id] = {
                        "path": result_path,
                        "caption": caption,
                        "db_embedding": torch.tensor(
                            image_query_results["embeddings"][0][i], dtype=torch.float32
                        ).to(device),
                    }

        print(
            f"Processing {len(candidate_data)} unique candidates from combined search."
        )
        final_combined_results_data = []  # List of (doc_path, combined_score, text_sim, image_sim)

        candidate_ids = list(candidate_data.keys())
        if candidate_ids:
            # Batch process all candidates
            all_db_embeddings_tensor = torch.stack(
                [candidate_data[p]["db_embedding"] for p in candidate_ids]
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
                candidate_id = candidate_ids[idx]
                data = candidate_data[candidate_id]
                doc_path = data["path"]
                caption = data["caption"]
                score = combined_scores_batch[idx].item()
                ts = text_sims_batch[idx].item()
                Is = image_sims_batch[idx].item()
                final_combined_results_data.append((doc_path, caption, score, ts, Is))

        final_combined_results_data.sort(key=lambda x: x[2], reverse=True)

        for doc_path, caption, score, ts, Is in final_combined_results_data[
            : int(initial_n_results_ui)
        ]:
            if os.path.exists(doc_path):
                gallery_images.append(
                    (doc_path, caption_with_score(caption, score, show_scores))
                )
                if verbose_ui:
                    print(
                        f"✓ {doc_path} (comb_score: {score:.4f}, txt_sim: {ts:.4f}, img_sim: {Is:.4f})"
                    )
            else:
                print(f"Missing: {doc_path}")

        if verbose_ui and not gallery_images and candidate_data:
            print("No candidates passed the combined similarity threshold.")
            for data_dict in candidate_data.values():
                doc_path = data_dict["path"]
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

    gallery_images = filter_gallery_by_active_folder(gallery_images, active_folder)

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
    current_db_path: str,
    active_folder: str = "All",
):
    """Returns a random selection of images from ChromaDB."""
    start_time = time.time()
    gallery_images = []

    if not current_chroma_client or not current_db_path:
        gr.Warning("Database not loaded. Please select a model.")
        return []

    try:
        current_chroma_client.get_collection("images")
    except Exception as e:
        print(f"Error getting collection: {e}")
        gr.Info(
            "Database collection 'images' not found or DB not initialized. "
            "Ensure `build-db.py` has been run for the selected model and then click 'Update/Sync Active DB'."
        )
        return []

    try:
        records = IndexStore(current_db_path).sample_media(
            int(initial_n_results_ui), active_folder
        )
        if not records:
            gr.Info("No images found in the database.")
            return []

        print(f"Random search: Sampling {len(records)} images from the index.")

        for record in records:
            doc_path = record.path
            if os.path.exists(doc_path):
                gallery_images.append((doc_path, record.relative_path))
            else:
                print(f"Missing: {doc_path}")

        gallery_images = filter_gallery_by_active_folder(gallery_images, active_folder)

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
        gr.Warning(f"Failed to retrieve random images: {e}", title="Error")
        return []


def clear_search_and_gallery():
    """Clears search inputs, gallery, duplicate state, and hides duplicate inspector."""
    return (
        "",
        None,
        [],
        [],
        gr.Tabs(selected="tab_search", elem_classes=["hide-tabs-nav"]),
        *render_duplicate_pair_view([], 0, ""),
    )


def get_allowed_gallery_paths() -> list[str]:
    """Returns paths Gradio may serve; broad access remains the default."""
    configured_paths = os.environ.get("LOCALLENS_ALLOWED_PATHS")
    if configured_paths:
        return [path for path in configured_paths.split(os.pathsep) if path]

    paths = ["/"]
    if os.name == "nt":
        paths = [f"{chr(d)}:\\" for d in range(ord("A"), ord("Z") + 1)]
    return paths


def also_store_gallery(fn, index: int = 0):
    """Repeats fn's gallery output (at `index`) as an extra output for gallery_items_state.

    The Gallery component only holds Gradio cache copies, so the original
    (path, caption) items are kept server-side for Show in Folder and trashing.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, tuple):
            return (*result, result[index])
        return result, result

    return wrapper


def handle_gallery_select(gallery_items: list, evt: gr.SelectData):
    """Remembers the selected result's original path for the Show in Folder button."""
    path = None
    if gallery_items and 0 <= evt.index < len(gallery_items):
        path = gallery_items[evt.index][0]
    return path, gr.Button(interactive=bool(path))


def handle_reveal(path: str | None):
    if not path or not reveal_in_file_manager(path):
        gr.Warning("Could not show the file in the file manager.")


def handle_reveal_duplicate(side: int, current_idx: int, pairs: list):
    """Reveals image A (side 1) or B (side 2) of the current duplicate pair."""
    if pairs and 0 <= current_idx < len(pairs):
        handle_reveal(pairs[current_idx][side])


css_path = Path(__file__).with_name("style.css")

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
        newContent.innerHTML = 'made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

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
        css_paths=str(css_path),
        title="LocalLens",
    ) as app:
        gr.Markdown("# LocalLens")

        active_model_path_state = gr.State(DEFAULT_MODEL_PATH)
        active_db_path_state = gr.State(None)

        model_state = gr.State(None)
        processor_state = gr.State(None)
        model_type_state = gr.State(None)
        logit_scale_state = gr.State(None)
        logit_bias_state = gr.State(None)
        chroma_client_state = gr.State(None)

        duplicate_pairs_state = gr.State([])
        selected_result_path_state = gr.State(None)
        gallery_items_state = gr.State([])
        duplicate_pair_idx_state = gr.State(0)

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
                    active_folder_dropdown = gr.Dropdown(
                        label="Active Folder",
                        choices=["All"],
                        value="All",
                        info="Limit searches to a specific indexed folder.",
                    )
                    # Newline-separated folder paths; parsed by handlers as the folder list.
                    indexed_folders_state = gr.State("")
                    index_summary = gr.Dataframe(
                        headers=["Folder", "Items"],
                        label="Indexed Folders",
                        interactive=False,
                        wrap=True,
                        column_widths=["75%", "25%"],
                    )
                    update_db_button = gr.Button(
                        "Update/Sync Full DB", variant="primary"
                    )

                    new_folder_textbox = gr.Textbox(
                        label="Folder Path to Add",
                        placeholder="/path/to/your/images",
                    )
                    add_folder_button = gr.Button("Add Folder", min_width=50)

                    delete_folder_dropdown = gr.Dropdown(
                        label="Folder to Delete",
                        choices=[],
                        value=None,
                        info="Removes the folder from the index; files on disk are untouched.",
                    )
                    delete_folder_button = gr.Button(
                        "Delete Folder", variant="stop", min_width=50
                    )

                gr.Markdown("### Find Duplicates")
                with gr.Accordion("Find Duplicates", open=False):
                    duplicate_folder_dropdown = gr.Dropdown(
                        label="Select Indexed Folder to Scan",
                        choices=["All"],
                        value="All",
                        info="Scan all indexed folders or select one folder.",
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
                        visible=False,
                        label="Logit Confidence Threshold (SigLIP)",
                        info="SigLIP model confidence. Higher values = more confident.",
                    )
                    clip_thresh_slider = gr.Slider(
                        minimum=0.0,
                        maximum=40.0,
                        step=0.1,
                        value=FALLBACK_CLIP_THRESHOLD,
                        visible=False,
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
                    show_scores_checkbox = gr.Checkbox(
                        value=False,
                        label="Show Scores in Captions",
                        info="Append each result's score (logit for text, cosine for image/combined) to its caption.",
                    )

            # Right Column: Tabs (Search Results & Duplicate Inspector)
            with gr.Column(scale=3):
                with gr.Tabs(
                    selected="tab_search",
                    elem_id="main_tabs",
                    elem_classes=["hide-tabs-nav"],
                ) as main_tabs:
                    with gr.Tab("Search Results", id="tab_search"):
                        results_gallery = gr.Gallery(
                            label="Search Results",
                            columns=3,
                            show_label=False,
                            elem_id="gallery",
                            interactive=False,
                        )
                        reveal_result_btn = gr.Button(
                            "Show in Folder", interactive=False, min_width=50
                        )

                    with gr.Tab("Duplicate Inspector", id="tab_duplicates"):
                        with gr.Column(visible=False) as duplicate_inspector_group:
                            with gr.Row(elem_classes=["dup-nav-row"]):
                                prev_pair_btn = gr.Button("Previous Pair", min_width=50)
                                pair_status_markdown = gr.Markdown(
                                    "Pair 0 of 0", elem_id="dup_status"
                                )
                                next_pair_btn = gr.Button("Next Pair", min_width=50)

                            with gr.Row():
                                with gr.Column():
                                    with gr.Group():
                                        img_a_preview = gr.Image(
                                            label="Image A",
                                            interactive=False,
                                            type="filepath",
                                            show_label=True,
                                        )
                                        img_a_info_html = gr.HTML(
                                            padding=False,
                                            elem_classes=["dup-info-html"],
                                        )
                                    reveal_a_btn = gr.Button(
                                        "Show in Folder", min_width=50
                                    )
                                    trash_a_btn = gr.Button(
                                        "Move to Recycle Bin",
                                        variant="stop",
                                        min_width=50,
                                    )
                                with gr.Column():
                                    with gr.Group():
                                        img_b_preview = gr.Image(
                                            label="Image B",
                                            interactive=False,
                                            type="filepath",
                                            show_label=True,
                                        )
                                        img_b_info_html = gr.HTML(
                                            padding=False,
                                            elem_classes=["dup-info-html"],
                                        )
                                    reveal_b_btn = gr.Button(
                                        "Show in Folder", min_width=50
                                    )
                                    trash_b_btn = gr.Button(
                                        "Move to Recycle Bin",
                                        variant="stop",
                                        min_width=50,
                                    )

        inspector_outputs = [
            duplicate_inspector_group,
            duplicate_pair_idx_state,
            pair_status_markdown,
            img_a_preview,
            img_a_info_html,
            img_b_preview,
            img_b_info_html,
            trash_a_btn,
            trash_b_btn,
            prev_pair_btn,
            next_pair_btn,
        ]
        inspector_and_gallery_outputs = [
            results_gallery,
            duplicate_pairs_state,
            main_tabs,
            *inspector_outputs,
            gallery_items_state,
        ]
        duplicate_view_reset_outputs = [
            duplicate_pairs_state,
            main_tabs,
            *inspector_outputs,
        ]

        def reset_duplicate_view():
            return (
                [],
                gr.Tabs(selected="tab_search", elem_classes=["hide-tabs-nav"]),
                *render_duplicate_pair_view([], 0, ""),
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
            indexed_folders_state,
            siglip_thresh_slider,
            clip_thresh_slider,
            combined_cosine_thresh_slider,
            image_only_cosine_thresh_slider,
            model_dropdown,
            batch_size_slider,
        ]

        def update_dropdown_from_indexed_folders(indexed_folders_text: str):
            """Update duplicate dropdown from indexed folders text."""
            if not indexed_folders_text:
                return gr.Dropdown(choices=[], value=None)
            folders = parse_indexed_folders_text(indexed_folders_text)
            if not folders:
                return gr.Dropdown(choices=[], value=None)
            choices = ["All"] + folders
            return gr.Dropdown(choices=choices, value="All")

        def update_all_folder_dropdowns(indexed_folders_text: str):
            """Update the active, duplicate, and delete folder dropdowns."""
            active_update = update_active_folder_dropdown(indexed_folders_text)
            dup_update = update_dropdown_from_indexed_folders(indexed_folders_text)
            delete_update = gr.Dropdown(
                choices=parse_indexed_folders_text(indexed_folders_text), value=None
            )
            return active_update, dup_update, delete_update

        folder_dropdowns = [
            active_folder_dropdown,
            duplicate_folder_dropdown,
            delete_folder_dropdown,
        ]

        refresh_index_summary = dict(
            fn=render_index_summary,
            inputs=[active_db_path_state],
            outputs=[index_summary],
        )

        app.load(
            fn=load_and_switch_model_db,
            inputs=[active_model_path_state],
            outputs=load_outputs,
            show_progress="full",
        ).then(
            fn=update_all_folder_dropdowns,
            inputs=[indexed_folders_state],
            outputs=folder_dropdowns,
        ).then(
            fn=also_store_gallery(empty_duplicate_results),
            inputs=[],
            outputs=inspector_and_gallery_outputs,
        ).then(**refresh_index_summary)

        model_dropdown.change(
            fn=load_and_switch_model_db,
            inputs=[model_dropdown],
            outputs=load_outputs,
            show_progress="full",
        ).then(
            fn=update_all_folder_dropdowns,
            inputs=[indexed_folders_state],
            outputs=folder_dropdowns,
        ).then(
            fn=also_store_gallery(empty_duplicate_results),
            inputs=[],
            outputs=inspector_and_gallery_outputs,
        ).then(**refresh_index_summary)

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
        ).then(**refresh_index_summary)

        def handle_add_folder_with_dropdown_update(*args):
            """Wrapper to update indexed folders display and all folder dropdowns."""
            indexed_folders_result = handle_add_folder_button_click(*args)
            return indexed_folders_result, *update_all_folder_dropdowns(
                indexed_folders_result
            )

        def handle_delete_folder_with_dropdown_update(*args):
            """Wrapper to update indexed folders display and all folder dropdowns."""
            indexed_folders_result = handle_delete_folder_button_click(*args)
            return indexed_folders_result, *update_all_folder_dropdowns(
                indexed_folders_result
            )

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
            outputs=[indexed_folders_state, *folder_dropdowns],
            show_progress="full",
        ).then(**refresh_index_summary)

        delete_folder_button.click(
            fn=handle_delete_folder_with_dropdown_update,
            inputs=[delete_folder_dropdown, active_db_path_state, chroma_client_state],
            outputs=[indexed_folders_state, *folder_dropdowns],
            show_progress="full",
        ).then(**refresh_index_summary)

        find_duplicates_button.click(
            fn=also_store_gallery(handle_find_duplicates),
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
                indexed_folders_state,
            ],
            outputs=inspector_and_gallery_outputs,
            show_progress="full",
        ).then(**refresh_index_summary)

        trash_inputs = [
            duplicate_pair_idx_state,
            duplicate_pairs_state,
            gallery_items_state,
            active_db_path_state,
            chroma_client_state,
            indexed_folders_state,
        ]

        trash_a_btn.click(
            fn=also_store_gallery(
                lambda idx, pairs, gal, db, client, text: handle_trash_duplicate(
                    "left", idx, pairs, gal, db, client, text
                )
            ),
            inputs=trash_inputs,
            outputs=inspector_and_gallery_outputs,
        ).then(**refresh_index_summary)

        trash_b_btn.click(
            fn=also_store_gallery(
                lambda idx, pairs, gal, db, client, text: handle_trash_duplicate(
                    "right", idx, pairs, gal, db, client, text
                )
            ),
            inputs=trash_inputs,
            outputs=inspector_and_gallery_outputs,
        ).then(**refresh_index_summary)

        results_gallery.select(
            fn=handle_gallery_select,
            inputs=[gallery_items_state],
            outputs=[selected_result_path_state, reveal_result_btn],
        )
        results_gallery.change(
            fn=lambda: (None, gr.Button(interactive=False)),
            inputs=None,
            outputs=[selected_result_path_state, reveal_result_btn],
        )
        reveal_result_btn.click(
            fn=handle_reveal, inputs=[selected_result_path_state], outputs=None
        )
        reveal_a_btn.click(
            fn=lambda idx, pairs: handle_reveal_duplicate(1, idx, pairs),
            inputs=[duplicate_pair_idx_state, duplicate_pairs_state],
            outputs=None,
        )
        reveal_b_btn.click(
            fn=lambda idx, pairs: handle_reveal_duplicate(2, idx, pairs),
            inputs=[duplicate_pair_idx_state, duplicate_pairs_state],
            outputs=None,
        )

        nav_inputs = [
            duplicate_pair_idx_state,
            duplicate_pairs_state,
            indexed_folders_state,
        ]

        prev_pair_btn.click(
            fn=lambda idx, pairs, text: handle_duplicate_nav(-1, idx, pairs, text),
            inputs=nav_inputs,
            outputs=inspector_outputs,
        )

        next_pair_btn.click(
            fn=lambda idx, pairs, text: handle_duplicate_nav(1, idx, pairs, text),
            inputs=nav_inputs,
            outputs=inspector_outputs,
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
            active_db_path_state,
            verbose_checkbox,
            active_folder_dropdown,
            show_scores_checkbox,
        ]

        query_textbox.submit(
            fn=reset_duplicate_view,
            inputs=[],
            outputs=duplicate_view_reset_outputs,
        ).then(
            fn=also_store_gallery(search),
            inputs=search_inputs,
            outputs=[results_gallery, gallery_items_state],
        )

        search_button.click(
            fn=reset_duplicate_view,
            inputs=[],
            outputs=duplicate_view_reset_outputs,
        ).then(
            fn=also_store_gallery(search),
            inputs=search_inputs,
            outputs=[results_gallery, gallery_items_state],
        )

        random_search_inputs = [
            initial_n_results_slider,
            chroma_client_state,
            active_db_path_state,
            active_folder_dropdown,
        ]
        feeling_lucky_button.click(
            fn=reset_duplicate_view,
            inputs=[],
            outputs=duplicate_view_reset_outputs,
        ).then(
            fn=also_store_gallery(random_search),
            inputs=random_search_inputs,
            outputs=[results_gallery, gallery_items_state],
        )

        clear_button.click(
            fn=also_store_gallery(clear_search_and_gallery, index=2),
            inputs=[],
            outputs=[
                query_textbox,
                query_image_input,
                results_gallery,
                duplicate_pairs_state,
                main_tabs,
                *inspector_outputs,
                gallery_items_state,
            ],
        )

    app.launch(inbrowser=True, allowed_paths=get_allowed_gallery_paths())

import os
import time

import chromadb
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from model_utils import load_model_and_processor
from build_db import db_add_folders, db_update_indexed_folders, db_delete_folder

# --- Configuration ---
AVAILABLE_MODELS = ["google/siglip2-so400m-patch16-512", "openai/clip-vit-large-patch14"]
DEFAULT_MODEL_PATH = None

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

INITIAL_N_RESULTS = 30  # Number of results to fetch from ChromaDB before applying threshold
SIGLIP_LOGIT_CONFIDENCE_THRESHOLD = -8.0  # Higher = more confident
CLIP_LOGIT_CONFIDENCE_THRESHOLD = 17.0  # Higher = more confident

verbose = False  # Set to True for detailed confidence logging


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
        if len(sub_parts) >= 2:
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


def load_and_switch_model_db(selected_model_path: str, progress=gr.Progress(track_tqdm=True)):
    """Loads the selected model, its processor, and initializes/loads its specific ChromaDB."""
    if not selected_model_path:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "No model selected. Please choose a model from the dropdown.",  # indexed_folders_display
        )
    progress(0, desc="Starting model...")

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

    print(f"Loading model: {selected_model_path}")

    try:
        progress(0.3, desc="Loading model...")
        new_model, new_processor, new_model_type = load_model_and_processor(selected_model_path, device, dtype)
        progress(0.6, desc="Calculating params...")

        new_logit_scale_exp_val = new_model.logit_scale.exp().item() if hasattr(new_model, "logit_scale") else None
        new_logit_bias_val = new_model.logit_bias.item() if hasattr(new_model, "logit_bias") else None
        progress(0.7, desc="Initializing ChromaDB client...")

        new_chroma_client = chromadb.PersistentClient(path=new_db_path)
        try:
            new_chroma_client.get_or_create_collection(name="images", metadata={"hnsw:space": "cosine"})
        except Exception as db_e:
            print(f"ChromaDB collection error: {db_e}")
            gr.Warning(f"ChromaDB client initialized for '{new_db_path}', but error ensuring collection: {db_e}")

        progress(0.9, desc="ChromaDB client initialized.")
        gr.Info(f"Successfully switched to model: {selected_model_path} and DB: {new_db_path}")

        current_indexed_folders = read_indexed_folders(new_db_path)
        return (
            selected_model_path,
            new_db_path,
            new_model,
            new_processor,
            new_model_type,
            new_logit_scale_exp_val,
            new_logit_bias_val,
            new_chroma_client,
            "\n".join(current_indexed_folders),
        )
    except Exception as e:
        print(f"Error loading model/DB: {e}")
        gr.Error(f"Error loading model/DB for {selected_model_path}: {e}")
        current_indexed_folders_on_error = read_indexed_folders(new_db_path)
        return (
            selected_model_path,
            new_db_path,
            None,
            None,
            None,
            None,
            None,
            None,
            "\n".join(current_indexed_folders_on_error),
        )


def handle_update_sync_button_click(
    current_model_path: str,
    current_db_path: str,
    active_model_state_val,
    active_processor_state_val,
    active_model_type_state_val,
    active_chroma_client_state_val,
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
        gr.Warning("Model components (model, processor, type, or DB client) not loaded. Please re-select a model.")
        return

    progress(0, desc="Starting DB update...")

    try:
        db_update_indexed_folders(
            db_path_str=current_db_path,
            collection_obj=active_chroma_client_state_val.get_collection(
                "images"
            ),
            model_obj=active_model_state_val,
            processor_obj=active_processor_state_val,
            device_str=device,
            model_type_str=active_model_type_state_val,
        )
        gr.Info(f"Database '{current_db_path}' update process completed successfully.")
        progress(1, desc="DB updated.")

    except Exception as e:
        print(f"Error updating DB directly: {e}")
        gr.Error(f"An unexpected error occurred while trying to update DB '{current_db_path}': {e}")
        progress(1, desc="Error.")


def handle_add_folder_button_click(
    new_folder_to_add: str,
    current_model_path: str,
    current_db_path: str,
    active_model_state_val,
    active_processor_state_val,
    active_model_type_state_val,
    active_chroma_client_state_val,
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
                folder_names_str = ", ".join([os.path.basename(f) for f in folders_being_processed])
            else:
                folder_names_str = "selected folder(s)"
            if total_images > 0:
                progress(0, desc=f"Processing {total_images} images from {folder_names_str}...")
            else:
                progress(0, desc=f"No images to process in {folder_names_str}.")
        elif status == "batch_processed":
            current_batch = kwargs.get("current_batch_num", 0)
            total_batches = kwargs.get("total_batches", 0)
            images_in_batch = kwargs.get("images_in_batch", 0)
            cumulative_processed = kwargs.get("cumulative_processed_this_run", 0)
            total_images_to_process = kwargs.get("total_images_to_process", 1)

            prog_fraction = cumulative_processed / total_images_to_process if total_images_to_process > 0 else 0
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
            gr.Info(f"Folder '{os.path.basename(new_folder_stripped)}' processed in {duration:.2f}s.")
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
        gr.Warning("Folder path to delete cannot be empty. Please enter a path from the list above.")
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
            gr.Info(f"Folder deleted successfully: {os.path.basename(folder_to_delete_stripped)}")
        else:
            gr.Warning(f"Folder not found: {os.path.basename(folder_to_delete_stripped)}")
        progress(1, desc="Delete folder completed.")

    except Exception as e:
        print(f"Error deleting folder: {e}")
        gr.Error(f"Failed to delete folder '{os.path.basename(folder_to_delete_stripped)}': {e}")
        progress(1, desc="Error.")

    current_indexed_folders = read_indexed_folders(current_db_path)
    return "\n".join(current_indexed_folders)


def search(
    query: str,
    initial_n_results_ui: int,
    logit_thresh_ui: float,
    clip_logit_thresh_ui: float,
    active_model_path: str,
    current_model,
    current_processor,
    current_model_type: str,
    current_logit_scale_exp_val: float,
    current_logit_bias_val: float,
    current_chroma_client,
):
    start_time = time.time()

    if not current_chroma_client or not current_model:
        gr.Warning("Model or Database not loaded. Please select a model.")
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

    processed_query = f"This is a photo of {query}."
    print(f"Searching: '{processed_query}'")

    with torch.no_grad():
        text_processing_args = {"text": processed_query, "return_tensors": "pt", "truncation": True}
        if current_model_type == "siglip":
            text_processing_args["padding"] = "max_length"
            text_processing_args["max_length"] = 64
        else:  # For CLIP
            text_processing_args["padding"] = True

        # Ensure max_length is not passed to CLIP if it was set for SigLIP
        if current_model_type != "siglip" and "max_length" in text_processing_args:
            del text_processing_args["max_length"]

        inputs = current_processor(**text_processing_args).to(device)
        text_features_from_model = current_model.get_text_features(**inputs)
        text_features_float32 = text_features_from_model.to(torch.float32)

        if current_model_type == "siglip" or current_model_type == "clip":
            text_emb_normalized_float32 = F.normalize(text_features_float32, p=2, dim=-1)
        else:
            text_emb_normalized_float32 = text_features_float32

    try:
        query_results = collection.query(
            query_embeddings=text_emb_normalized_float32.cpu().squeeze(0).tolist(),
            n_results=int(initial_n_results_ui),
            include=["documents", "embeddings", "distances"],
        )
    except Exception as e:
        print(f"ChromaDB query error: {e}")
        gr.Warning(f"Error searching images: {e}")
        return []

    gallery_images = []
    if query_results["documents"] and query_results["embeddings"] and query_results["documents"][0]:
        candidates_count = len(query_results["documents"][0])
        print(f"Processing {candidates_count} candidates from DB")

        for doc_path, emb_list, _ in zip(
            query_results["documents"][0], query_results["embeddings"][0], query_results["distances"][0]
        ):
            passes_threshold = False
            logit_val_for_print = 0

            if current_model_type == "siglip":
                img_emb_candidate_float32 = torch.tensor(emb_list, dtype=torch.float32).to(device)
                cos_sim = F.cosine_similarity(
                    text_emb_normalized_float32.squeeze(0), img_emb_candidate_float32, dim=0
                ).item()
                if current_logit_scale_exp_val is not None and current_logit_bias_val is not None:
                    siglip_logit = (cos_sim * current_logit_scale_exp_val) + current_logit_bias_val
                    if siglip_logit > logit_thresh_ui:
                        passes_threshold = True
                    logit_val_for_print = siglip_logit
                else:
                    print("Warning: SigLIP logit scale/bias unavailable for model {active_model_path}")

            elif current_model_type == "clip":
                img_emb_candidate_float32 = torch.tensor(emb_list, dtype=torch.float32).to(device)
                cos_sim = F.cosine_similarity(
                    text_emb_normalized_float32.squeeze(0), img_emb_candidate_float32, dim=0
                ).item()
                if current_logit_scale_exp_val is not None:
                    clip_scaled_logit = cos_sim * current_logit_scale_exp_val
                    if clip_scaled_logit > clip_logit_thresh_ui:
                        passes_threshold = True
                    logit_val_for_print = clip_scaled_logit
                else:
                    print(f"Warning: CLIP logit scale unavailable for model {active_model_path}")

            if passes_threshold:
                try:
                    img = Image.open(doc_path)
                    gallery_images.append((img, doc_path))
                    if verbose:
                        print(f"✓ {doc_path} (cos: {cos_sim:.4f}, logit: {logit_val_for_print:.4f})")
                except FileNotFoundError:
                    print(f"Missing: {doc_path}")
                except Exception as e:
                    print(f"Error opening {doc_path}: {e}")
            else:
                if verbose:
                    print(f"✗ {doc_path} (cos: {cos_sim:.4f}, logit: {logit_val_for_print:.4f})")

    result_count = len(gallery_images)
    if result_count == 0:
        print("No images found above confidence threshold")
    else:
        print(f"Returning {result_count} images")

    end_time = time.time()
    print(f"Search completed in {(end_time - start_time):.2f}s")

    return gallery_images


def clear_search_and_gallery():
    """Clears the search query textbox and the results gallery."""
    return "", []


custom_css = """
#gallery .grid-wrap {
    max-height: 900px !important;
}
"""

if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Default(primary_hue="purple"), css=custom_css, title="Where's My Pic?") as app:
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
                    placeholder="This is a photo of...",
                    show_label=False,
                )
                with gr.Row():
                    search_button = gr.Button("Search", variant="primary", min_width=50)
                    clear_button = gr.Button("Clear", min_width=50)

                gr.Markdown("### Model & Database Management")
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=None,
                    label="Select Model",
                )

                gr.Markdown("#### Indexed Folders for Active DB")
                indexed_folders_display = gr.Textbox(
                    label="Currently Indexed Folders", interactive=False, lines=3, max_lines=5
                )
                update_db_button = gr.Button("Full Update/Sync Active DB", variant="primary")

                new_folder_textbox = gr.Textbox(
                    label="Folder Path to Add/Delete",
                    placeholder="/path/to/your/images",
                )
                with gr.Row():
                    add_folder_button = gr.Button("Add Folder", min_width=50)
                    delete_folder_button = gr.Button("Delete Folder", variant="stop", min_width=50)

                gr.Markdown("### Search Parameters")
                with gr.Accordion("Search Parameters", open=False):
                    initial_n_results_slider = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=INITIAL_N_RESULTS,
                        label="Initial N Results (Max Images to Fetch)",
                    )
                    siglip_thresh_slider = gr.Slider(
                        minimum=-20.0,
                        maximum=0.0,
                        step=0.1,
                        value=SIGLIP_LOGIT_CONFIDENCE_THRESHOLD,
                        label="Logit Confidence Threshold (SigLIP)",
                    )
                    clip_thresh_slider = gr.Slider(
                        minimum=0.0,
                        maximum=30.0,
                        step=0.1,
                        value=CLIP_LOGIT_CONFIDENCE_THRESHOLD,
                        label="Logit Confidence Threshold (CLIP)",
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
        ]

        app.load(
            fn=load_and_switch_model_db,
            inputs=[active_model_path_state],
            outputs=load_outputs,
            show_progress="full",
        )

        model_dropdown.change(
            fn=load_and_switch_model_db, inputs=[model_dropdown], outputs=load_outputs, show_progress="full"
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
            ],
            outputs=[],
            show_progress="full",
        )

        add_folder_button.click(
            fn=handle_add_folder_button_click,
            inputs=[
                new_folder_textbox,
                active_model_path_state,
                active_db_path_state,
                model_state,
                processor_state,
                model_type_state,
                chroma_client_state,
            ],
            outputs=[indexed_folders_display],
            show_progress="full",
        )

        delete_folder_button.click(
            fn=handle_delete_folder_button_click,
            inputs=[new_folder_textbox, active_db_path_state, chroma_client_state],
            outputs=[indexed_folders_display],
            show_progress="full",
        )

        search_inputs = [
            query_textbox,
            initial_n_results_slider,
            siglip_thresh_slider,
            clip_thresh_slider,
            active_model_path_state,  # For context if needed by search (e.g. logging)
            model_state,
            processor_state,
            model_type_state,
            logit_scale_state,
            logit_bias_state,
            chroma_client_state,
        ]

        query_textbox.submit(fn=search, inputs=search_inputs, outputs=[results_gallery])
        search_button.click(fn=search, inputs=search_inputs, outputs=[results_gallery])

        clear_button.click(fn=clear_search_and_gallery, inputs=[], outputs=[query_textbox, results_gallery])

    app.launch(inbrowser=True)

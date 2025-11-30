## Local Lens

A local image search engine powered by CLIP/SigLIP 2. Search for images in your local directories using natural language queries.

## Features

- Image-text retrieval using CLIP/SigLIP 2/Other models
- Index and update local image directories using ChromaDB
- Perform natural language searches to find relevant images
- Find duplicate images in a ChromaDB database
- Web UI (Gradio)

## Requirements

- Python 3.10+
- PyTorch (CPU, CUDA, ROCm)

## Install

### Windows Portable

Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/LocalLens/releases)

- Pre-downloaded package: Download per version, no setup required, and no included update script. Contains `PyTorch v2.9.1+cu128`.

### Manual install

1. Clone and enter the repo

```bash
git clone https://github.com/meangrinch/LocalLens.git
cd LocalLens
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install PyTorch (see: [PyTorch Install](https://pytorch.org/get-started/locally/))

```bash
# Example (CUDA 12.8)
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# Example (CPU)
pip install torch
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Web UI (Gradio)

1. Select a model (automatically downloads to hugging face cache `~/.cache/huggingface/hub`)
2. Add image directories to your Chroma database (via the "Database Management" dropdown in the UI, or via CLI)
3. Enter your search query (e.g., "an orange and black butterfly") and/or upload an image (for reverse image search/combined text + image search)
4. The application will display the results in order of confidence
5. Update/sync indexed directories if necessary

### Find duplicates

Click "Find Duplicates" in the UI with a specified indexed image directory to return similar matching images pairs.

## Updating

- Windows portable:
  - Pre-downloaded Package: Download the latest version from the releases page
- Manual install: from the repo root:

```bash
git pull
```

## License & credits

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)
- Inspired by [Where's My Pic?](https://github.com/Om-Alve/Wheres_My_Pic) by [@Om-Alve](https://github.com/Om-Alve)

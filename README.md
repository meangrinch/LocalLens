## Local Lens

A local image search engine powered by CLIP/SigLIP. Search for images in your local directories using natural language queries.

## Features
- Image-text retrieval using CLIP/SigLIP/Other models.
- Index and update local image directories using ChromaDB.
- Perform natural language searches to find relevant images.
- Find duplicate images in a ChromaDB database.
- Web Interface (Gradio).

## Requirements
- Python 3.10-3.12

## Install

### Windows Portable

Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/LocalLens/releases)

### Manual install
1) Clone and enter the repo
```bash
git clone https://github.com/meangrinch/LocalLens.git
cd LocalLens
```
2) Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```
3) Install PyTorch (see: [PyTorch Install](https://pytorch.org/get-started/locally/))
```bash
# Example (CUDA 12.8)
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# Example (CPU)
pip install torch
```
4) Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Web UI (Gradio)
1) Add image directories to your Chroma database (via the "Database Management" dropdown in the UI, or via CLI)
2) Enter your search query (e.g., "an orange and black butterfly") and/or upload an image
3) The application will display the results in order of confidence
4) Update/sync indexed directories if necessary

### Find duplicates
Run `find_duplicates.py` with a specified image directory to return similar matching images:
```bash
# Activate venv and run script
.\venv\Scripts\activate
python find_duplicates.py path/to/image/directory

# Full options
python find_duplicates.py --help
```

## Updating
- Manual install: from the repo root:
```bash
git pull
```

## License & credits
- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)
- Inspired by [Where's My Pic?](https://github.com/Om-Alve/Wheres_My_Pic) by [@Om-Alve](https://github.com/Om-Alve)
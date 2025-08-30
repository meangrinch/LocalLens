## Local Lens

A local image search engine powered by CLIP/SigLIP. Search for images in your local directories using natural language queries.

## Features
- Image-text retrieval using CLIP/SigLIP.
- Index and update local image directories using ChromaDB.
- Perform natural language searches to find relevant images.
- Web Interface (Gradio).

## Requirements
- Python 3.10-3.12

## Install

### Windows Portable

Download the standalone zip (NVIDIA GPU or CPU) from the [releases page](https://github.com/meangrinch/LocalLens/releases). 

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
# Example (CUDA 12.4)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
# Example (CPU)
pip install torch
```
4) Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1) Add image directories to your Chroma database (via the "Database Management" dropdown in the UI, or via CLI)
2) Enter your search query (e.g., "an orange and black butterfly") and/or upload an image
3) The application will display the results in order of confidence
4) Update/sync indexed directories if necessary
5) (Optional) Edit and run `run_find_duplicates.bat` with a specified image directory to return similar matching images

## Updating
```bash
git pull
```

## License & credits
- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)
- Inspired by [Where's My Pic?](https://github.com/Om-Alve/Wheres_My_Pic) by [@Om-Alve](https://github.com/Om-Alve)
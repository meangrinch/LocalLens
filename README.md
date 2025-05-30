# Local Lens

A local image search engine powered by CLIP/SigLIP. Search for images in your local directories using natural language queries.

## Features

*   Image-text retrieval using CLIP/SigLIP.
*   Index and update local image directories using ChromaDB.
*   Perform natural language searches to find relevant images.
*   Web Interface (Gradio).

## Requirements

*   Python 3.10-3.12

## Installation

### Windows Portable

Download the standalone zip (NVIDIA GPU or CPU) from the [releases page](https://github.com/meangrinch/Local_Lens/releases). 

### Manual Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/meangrinch/LocalLens.git
    cd Wheres_My_Pic
    ```

2. **Create Virtual Environment (Recommended):**
    ```bash
    # Create venv
    python -m venv venv
    
    # Activate (Windows CMD/PowerShell)
    .\venv\Scripts\activate
    
    # Activate (Linux/macOS/Git Bash)
    source venv/bin/activate
    ```

3.  **Install PyTorch:**
    ```python
    # Example (CUDA 12.4)
    pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

    # Example (CPU)
    pip install torch
    ```
    *Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for system-specific commands.*

4.  **Install Dependencies:**
    ```python
    pip install -r requirements.txt
    ```

## Usage

1.  Add image directories to your Chroma database.

2.  Enter your search query (e.g., "an orange and black butterfly").

3.  The application will display the results in order of confidence.

4.  Update/sync indexed directories after modifying the original.

## Updating

Navigate to the `LocalLens` directory and run:
```bash
git pull
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Credits

Inspired by [Where's My Pic?](https://github.com/Om-Alve/Wheres_My_Pic) by @Om-Alve.

## Author

[grinnch](https://github.com/meangrinch)
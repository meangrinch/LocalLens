<p align="center">
  <a href="../../../README.md">English</a> |
  <a href="README.md">简体中文</a>
</p>

<h1 align="center"><b>LocalLens</b></h1>

<p align="center">
  <img src="https://img.shields.io/github/v/release/meangrinch/LocalLens?label=Release&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/downloads/meangrinch/LocalLens/total?label=Downloads&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/license/meangrinch/LocalLens?labelColor=181717&color=2ea44f" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&labelColor=181717" />
</p>

<div align="center">
适用于本地图像与视频收藏的离线视觉搜索引擎。使用视觉嵌入模型将目录索引至 ChromaDB，支持自然语言查询与以图搜图，并附带重复文件检测功能。
</div>

<br/>

<div align="center">
  <img src="../../images/example_screenshot.png" alt="LocalLens UI 截图" width="1000" />
</div>

---

## 快速上手

### 1. 下载

- **便携版（推荐）：** 从 [Releases](https://github.com/meangrinch/LocalLens/releases/tag/portable) 下载便携版构建。
  - Windows： 无需额外系统要求。
  - Linux / macOS： 需要系统安装 Python 3.10+ 和 Git。
- **源码安装：**
  ```bash
  git clone https://github.com/meangrinch/LocalLens.git
  cd LocalLens
  python -m venv venv
  .\venv\Scripts\activate # 或 source venv/bin/activate (Linux/macOS)
  pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130 # 或 pip install torch==2.12.0 torchvision==0.27.0 (macOS)
  pip install -r requirements.txt
  ```

*更多信息请参阅 [安装](INSTALLATION.md)。*

### 2. 索引

- **Web UI：** 在 Database Management 中输入目录路径，然后点击 Add Folder。
- **CLI：**
  ```bash
  python build_db.py --model_path "google/siglip2-so400m-patch16-512" --db_path "img_db/siglip2_so400m" --add "path/to/images"
  ```
- **同步更改：** 在 Web UI 中点击 Update/Sync，或在命令行中传入 `--update`，重新扫描已索引文件夹中的新增或删除内容。

### 3. 搜索

- **文本搜索：** 输入描述（例如 "an orange and black butterfly"）即可按相似度排序检索匹配的媒体。
- **以图搜图：** 上传图像以查找视觉上相似的文件，或结合文本与图像输入精细调整搜索结果。
- **查找重复项：** 在 Find Duplicates 选项卡中选择已索引的文件夹并设置相似度阈值，以找出重复的图像与视频对。
- **访问限制（可选）：** 默认情况下，Gradio 相册可以显示来自任何已索引目录的图像。在启动应用前于系统环境中设置 `LOCALLENS_ALLOWED_PATHS` 即可限制允许访问的目录。

---

## 功能特点

- **索引**：递归目录索引与增量同步（ChromaDB）
- **搜索**：自然语言文本、图像以及图文混合查询
- **以图搜图**：通过上传的图像查找视觉上相似的媒体
- **重复检测**：通过相似度阈值定位完全重复与高度相似的图像与视频对
- **模型**：视觉嵌入骨干网络（SigLIP 2、MetaCLIP、DFN-CLIP、CLIP）
- **格式**：支持常见图像与视频格式（jpg、png、webp、mp4、mkv 等）
- **界面**：Web UI (Gradio) 和命令行界面 (CLI)

---

## 文档

- [硬件要求](HARDWARE_REQUIREMENTS.md)
- [安装](INSTALLATION.md)

---

## 支持项目

LocalLens 是免费且开源的。如果它为您查找照片或清理重复图像节省了时间，欢迎考虑支持它的开发！

<p align="center">
  <a href="https://ko-fi.com/grinnch" target="_blank">
    <img src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" alt="Support on Ko-fi" height="38"/>
  </a>
</p>

---

## 许可证和鸣谢

- 许可证：Apache-2.0（参见 [LICENSE](../../../LICENSE)）
- 作者：[grinnch](https://github.com/meangrinch)
- 灵感来源：[Om-Alve](https://github.com/Om-Alve) 的 [Where's My Pic?](https://github.com/Om-Alve/Wheres_My_Pic)

<details>
<summary><b>ML 模型与相关开源库</b></summary>

- SigLIP 2 SO400M (384): [Google](https://huggingface.co/google/siglip2-so400m-patch14-384)
- SigLIP 2 SO400M (512): [Google](https://huggingface.co/google/siglip2-so400m-patch16-512)
- SigLIP 2 Giant: [Google](https://huggingface.co/google/siglip2-giant-opt-patch16-384)
- DFN5B CLIP ViT-H-14: [Apple](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378)
- MetaCLIP ViT-H-14: [Meta AI](https://huggingface.co/facebook/metaclip-h14-fullcc2.5b)
- LAION-CLIP ViT-H-14: [LAION](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
- OpenAI CLIP ViT-L-14: [OpenAI](https://huggingface.co/openai/clip-vit-large-patch14)
- SigLIP 2 Base: [Google](https://huggingface.co/google/siglip2-base-patch16-224)
- ChromaDB: [Chroma](https://github.com/chroma-core/chroma)

</details>

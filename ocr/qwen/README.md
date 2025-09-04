# [Qwen-VL](https://github.com/QwenLM/Qwen-VL)

## Overview
This OCR pipeline converts PDF documents to text using the Qwen2.5-VL-3B-Instruct vision-language model, which can read and understand text in images including Greek and English text while preserving original formatting and structure.

## In-Depth Processing
### Image Processing
The image processing pipeline prepares PDF content for the vision-language model:
**PDF Conversion:**
- Converts PDF pages to high-resolution images (configurable DPI, default 300)
- Maintains original document layout and formatting
- Preserves color information for optimal model performance

### Text Extraction
The text extraction leverages the vision-language model's capabilities:
**Vision-Language Processing:**
- Processes images through Qwen2.5-VL-3B-Instruct model
- Handles Greek and English text simultaneously without language detection
- Maintains spatial relationships and original formatting
- Uses conversation-based prompting for accurate transcription
- Preserves diacritics, accents, and special characters
- No translation or interpretation - pure transcription of visible text

## Output
This tool processes each PDF page and produces a single consolidated output file:
1. **Page-Separated Text** (`_qwen_vlm.txt`)  
   Contains the complete transcribed text with clear page separators (`--- Page N ---`)

## Installation and Setup
### Prerequisites
**System Dependencies:**
```bash
sudo apt update
sudo apt install poppler-utils
```
**Python Dependencies:**
```bash
pip install torch transformers pdf2image Pillow tqdm qwen-vl-utils
```
**GPU Requirements:**
- NVIDIA GPU with CUDA support
- Minimum 6GB VRAM (8GB+ recommended)
- The script automatically downloads the Qwen2.5-VL-3B-Instruct model (~6.5GB) on first run

### How to Run the OCR
**Basic Usage:**
```bash
python qwen_vlm_ocr.py <document>.pdf
```
**With Custom DPI:**
```bash
python qwen_vlm_ocr.py <document>.pdf --dpi 600
```
**With Custom Output Path:**
```bash
python qwen_vlm_ocr.py <document>.pdf --output /path/to/output.txt
```
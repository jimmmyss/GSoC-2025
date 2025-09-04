# DOTS OCR
## Overview
This OCR pipeline converts PDF documents to text using the DOTS OCR model, a specialized 1.7B parameter model for optical character recognition that preserves document layout and structure while handling complex formatting elements.

## In-Depth Processing
### Model Loading
The model loading process handles component initialization:
**Component Setup:**
- Loads DOTS OCR model with flash attention optimization
- Uses bfloat16 precision for efficient processing
- Separates tokenizer and image processor loading for compatibility
- Handles processor compatibility issues with simplified approach

### Text Extraction
The extraction process uses comprehensive OCR prompting:
**Document Analysis:**
- Processes images through DOTS OCR 1.7B parameter model
- Uses detailed instruction prompting for comprehensive text extraction
- Maintains original formatting, spacing, and document structure
- Preserves reading order (top to bottom, left to right)
- Handles mathematical equations, formulas, and special symbols
- Preserves table structures with appropriate formatting
- Indicates column breaks in multi-column layouts
- Marks unclear or partially visible text with appropriate tags
- No interpretation or correction - pure transcription with structure preservation

## Output
This tool processes PDF pages and produces comprehensive output:
1. **Plain Text** (`_dots_working.txt`)  
   Contains the extracted text with preserved formatting and structure
2. **Raw JSON Response** (`_dots_working_raw.json`)  
   Complete results with metadata, timing information, and processing details

## Installation and Setup
### Prerequisites
**System Dependencies:**
```bash
sudo apt update
sudo apt install python3-pip
```
**Python Dependencies:**
```bash
pip install torch transformers PyMuPDF Pillow qwen-vl-utils
```
**Model Requirements:**
- Local DOTS OCR model installation at specified path
- NVIDIA GPU with CUDA support (recommended)
- Flash attention support for optimized performance
- The model requires approximately 3.5GB of storage space

### How to Run the OCR
**Basic Usage:**
```bash
python dots_ocr_working.py
```

**Note:** The current implementation processes a hardcoded PDF path. Modify the `pdf_path` variable in the script to process different documents.
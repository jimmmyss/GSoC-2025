# Nanonets OCR
## Overview
This OCR pipeline converts PDF documents to text using the Nanonets-OCR-s model from Hugging Face, which specializes in document understanding and can extract text while preserving formatting, handling mathematical expressions, tables, and other document elements.

## In-Depth Processing
### PDF Conversion
The PDF processing pipeline converts documents to high-quality images:
**Image Generation:**
- Converts PDF pages to images using PyMuPDF at 300 DPI
- Maintains original document quality and layout
- Preserves color information for optimal model performance

### Text Extraction
The text extraction uses the specialized Nanonets model:
**Document Understanding:**
- Processes images through Nanonets-OCR-s model optimized for document analysis
- Uses conversation-based prompting for comprehensive text extraction
- Handles mathematical expressions and converts them to LaTeX format
- Extracts tables and formats them as HTML
- Describes images with descriptive alt text in `<img>` tags
- Identifies signatures and wraps them in `<signature>` tags
- Preserves document structure with markdown formatting for headers, lists, and code blocks
- No interpretation or assumptions - pure transcription with enhanced formatting

## Output
This tool processes each PDF page and produces multiple output formats:
1. **Raw JSON Response** (`_nanonets_raw.json`)  
   Contains complete OCR results with metadata and processing times
2. **Plain Text** (`_nanonets.txt`)  
   The extracted text content with page separators
3. **Markdown Format** (`_nanonets.md`)  
   Formatted text with proper markdown structure and document metadata

## Installation and Setup
### Prerequisites
**System Dependencies:**
```bash
sudo apt update
sudo apt install python3-pip
```
**Python Dependencies:**
```bash
pip install torch transformers PyMuPDF Pillow tqdm
```
**GPU Requirements:**
- NVIDIA GPU with CUDA support (optional but recommended)
- Minimum 8GB VRAM for optimal performance
- The script automatically downloads the Nanonets-OCR-s model (~7.5GB) on first run

### How to Run the OCR
**Basic Usage:**
```bash
python nanonets_ocr.py <document>.pdf
```
**Multiple PDFs:**
```bash
python nanonets_ocr.py doc1.pdf doc2.pdf doc3.pdf
```
**With Custom Output Directory:**
```bash
python nanonets_ocr.py <document>.pdf --output-dir /path/to/output
```
**Force CPU Usage:**
```bash
python nanonets_ocr.py <document>.pdf --device cpu
```
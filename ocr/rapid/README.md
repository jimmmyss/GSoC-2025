# Greek PDF OCR with Docling + RapidOCR

## Overview

This OCR pipeline converts Greek academic PDFs to structured text using Docling's layout analysis combined with RapidOCR's ONNX runtime for high-accuracy Greek text recognition. The tool performs sophisticated document processing with GPU acceleration:

1. **Layout Analysis** - Uses Docling's deep learning models to understand document structure
2. **Greek OCR Processing** - Employs PP-OCRv5 ONNX models specifically trained for Greek text
3. **Text Normalization** - Applies Unicode normalization and cleanup for mathematical content
4. **Structured Export** - Outputs both Markdown and JSON formats with preserved structure

## Processing Pipeline

### Document Analysis

**Layout Detection:**
- GPU-accelerated deep learning models for document layout understanding
- Identifies text blocks, headers, tables, formulas, and code sections
- Preserves reading order and hierarchical structure
- Handles complex multi-column academic layouts

**OCR Processing:**
- Forces full-page OCR (ignores embedded PDF text for consistency)
- Uses PP-OCRv5 Greek recognition models via ONNX runtime
- Supports both Greek (el) and English (en) languages simultaneously
- Applies configurable image scaling (default 1.25x) for improved recognition
- Implements text orientation classification for rotated content

### Advanced Features

**Formula Recognition (Optional):**
- Integrates Docling's CodeFormula model for mathematical expressions
- GPU-accelerated formula parsing and LaTeX conversion
- Configurable batch processing for performance optimization

**Text Normalization:**
- Unicode NFC normalization for consistent character representation
- Removes zero-width characters and formatting artifacts
- Handles complex Greek polytonic and mathematical Unicode ranges

## Output

The tool produces comprehensive output files:

1. **Structured Markdown** (`{filename}.md`) - Clean, readable text with preserved structure
2. **Structured JSON** (`{filename}.json`) - Complete document data with metadata
3. **Processing Metrics** (`{filename}.metrics.json`) - Detailed timing information
4. **Per-Page Metrics** (`{filename}.per_page.metrics.json`) - Page-by-page statistics

## Installation and Setup

### Prerequisites

**System Requirements:**
- NVIDIA GPU with CUDA support (required)
- CUDA-compatible drivers and runtime
- Python 3.8+ with pip

**Dependencies:**
```bash
pip install -r requirements.txt
```

### Required Model Files

You need these ONNX models:

1. **Detection Model** - PP-OCRv5 text detection (`inference.onnx`)
2. **Recognition Model** - PP-OCRv5 Greek text recognition (`inference.onnx`) 
3. **Classification Model** - Text orientation classifier (`ch_ppocr_mobile_v2.0_cls_infer.onnx`)
4. **Recognition Keys** - Greek character dictionary (`greek_ppocrv5_keys.txt`)

## How to Run

### Basic Usage

```bash
python greek_pdf_ocr.py INPUT_DIR OUTPUT_DIR \
  --onnx-det /path/to/det_onnx/inference.onnx \
  --onnx-rec /path/to/rec_onnx/inference.onnx \
  --onnx-cls /path/to/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-keys /path/to/greek_ppocrv5_keys.txt
```

### Using the Convenience Script

```bash
./scripts/run_onnx.sh \
  --det /path/to/det_onnx/inference.onnx \
  --rec /path/to/rec_onnx/inference.onnx \
  --keys /path/to/greek_ppocrv5_keys.txt \
  --in /path/to/pdf/directory \
  --out /path/to/output/directory
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `cuda:0` | GPU device for processing |
| `--text-score` | `0.50` | OCR confidence threshold |
| `--images-scale` | `1.25` | Image scaling factor |
| `--docling-formula` | `False` | Enable formula recognition |
| `--normalize-output` | `True` | Apply Unicode normalization |

## Files to Upload to GitHub

### Essential Files
1. `greek_pdf_ocr.py` - Main OCR script
2. `requirements.txt` - Python dependencies  
3. `README.md` - This documentation
4. `scripts/run_onnx.sh` - Convenience wrapper script
5. `MODELS.md` - Model setup guide

### Model Files (Consider Git LFS)
6. `models/det_onnx/inference.onnx` - Detection model (~50-100MB)
7. `models/rec_onnx/inference.onnx` - Greek recognition model (~50-100MB)
8. `models/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx` - Classifier (~10MB)
9. `models/keys/greek_ppocrv5_keys.txt` - Character dictionary (~10KB)

**Note**: ONNX model files are large (50-200MB each). Consider using Git LFS or providing download instructions instead of direct repository inclusion.
#[RapidOCR](https://github.com/RapidAI/RapidOCR) & [https://github.com/onnx/onnx])(onnx)

## Overview

This OCR pipeline converts PDF documents to structured text using Docling's advanced layout analysis combined with RapidOCR's ONNX-based OCR engine, specifically optimized for Greek and English academic documents. The tool performs four main operations:

1. **Document Layout Analysis** - Uses Docling's GPU-accelerated models to detect and classify document elements (text blocks, tables, formulas, etc.)
2. **Advanced OCR Processing** - Employs RapidOCR with PP-OCRv5 Greek models via ONNX Runtime for high-accuracy text extraction
3. **Formula & Code Enrichment** - Optional mathematical formula and code block recognition using specialized neural models
4. **Multi-Format Export** - Outputs both structured Markdown and detailed JSON with comprehensive metadata

## In-Depth Processing

### Preprocessing

**Layout Analysis:**
- GPU-accelerated document structure detection using Docling's deep learning models
- Intelligent text region identification and classification
- Table structure recognition with TableFormer models
- Automatic reading order determination for complex layouts

**Image Enhancement:**
- Configurable raster scaling (default 1.25x) before OCR to sharpen thin glyphs
- Automatic orientation detection and correction using classification models
- Optimized preprocessing specifically for math-heavy Greek academic content

**Text Detection & Recognition:**
- PP-OCRv5 detection model for precise text region localization
- Greek-specific PP-OCRv5 recognition model with custom character dictionary
- Dual-language support (Greek + English) with automatic language detection
- Configurable confidence thresholds (default 0.50) for quality control

### Post-Processing

**Content Structure Preservation:**
- Maintains document hierarchy and reading order from layout analysis
- Preserves table structures with cell-level accuracy
- Handles complex multi-column layouts common in academic papers

**Mathematical Content Processing:**
- Optional CodeFormula model for LaTeX mathematical expression recognition
- Specialized processing for Greek mathematical notation and symbols
- Formula truncation and sanitization to prevent excessive whitespace runs

**Unicode Normalization:**
- NFC normalization for stable Unicode representation
- Zero-width character removal (ZWSP, ZWNJ, ZWD, BOM)
- Consistent handling of Greek diacritics and mathematical symbols

## Output

This tool processes academic PDFs and produces comprehensive output files optimized for analysis and further processing:

1. **Structured Markdown** (`{filename}.md`)  
   Clean, hierarchically organized text preserving document structure, tables, and reading order

2. **Detailed JSON** (`{filename}.json`)  
   Complete document metadata including bounding boxes, confidence scores, and element classifications

3. **Performance Metrics** (`{filename}.metrics.json`)  
   Detailed timing analysis for each processing stage (OCR, layout, parsing, enrichment)

4. **Per-Page Analytics** (`{filename}.per_page.metrics.json`)  
   Page-level breakdown including formula counts, processing times, and content statistics

## Installation and Setup

### Prerequisites

**System Requirements:**
- NVIDIA GPU with CUDA support (required for optimal performance)
- CUDA-compatible drivers (NVIDIA 470+ recommended)
- Python 3.10+ 

**Core Dependencies:**
```bash
pip install docling[rapidocr]==2.48.0
pip install rapidocr_onnxruntime==1.4.4
pip install onnxruntime-gpu==1.18.1
pip install "numpy<2"
pip install pyyaml>=6.0
pip install tqdm>=4.67
```

**Important:** Ensure only `onnxruntime-gpu` is installed, not the CPU version:
```bash
pip uninstall -y onnxruntime  # Remove CPU version if present
```

**ONNX Model Files Required:**
You need three ONNX models for the pipeline:
- **Detection Model**: PP-OCRv5 detection ONNX (`inference.onnx`)
- **Recognition Model**: PP-OCRv5 Greek recognition ONNX (`inference.onnx`) 
- **Classification Model**: Text orientation classifier (auto-located from RapidOCR installation)

**Greek Character Dictionary:**
Generate from your Greek Paddle model's `inference.yml`:
```bash
python scripts/extract_keys.py --in-yml /path/to/inference.yml --out greek_keys.txt
```

### Critical Setup Step: Docling Patch

Due to a parameter mapping issue in Docling 2.48.0, apply this one-line patch:

**Location:** `.venv/lib/python3.10/site-packages/docling/models/rapid_ocr_model.py`  
**Change:** Line containing `"Rec.keys_path"` → `"Rec.rec_keys_path"`

Or use the automated patch script:
```bash
bash scripts/repatch_docling.sh
```

### Verification

**Check GPU Providers:**
```bash
python -c "import onnxruntime as ort; print('CUDAExecutionProvider' in ort.get_available_providers())"
```

## How to Run the OCR

### Basic Usage

**Process PDFs with ONNX backend:**
```bash
python rapid.py input_pdfs/ output_dir/ \
  --onnx-det path/to/det/inference.onnx \
  --onnx-rec path/to/rec/inference.onnx \
  --onnx-cls path/to/cls/inference.onnx \
  --rec-keys path/to/greek_keys.txt
```

### Advanced Options

**GPU Configuration:**
```bash
python rapid.py input_pdfs/ output_dir/ \
  --device cuda:0 \
  --onnx-det det/inference.onnx \
  --onnx-rec rec/inference.onnx \
  --onnx-cls cls/inference.onnx \
  --rec-keys greek_keys.txt
```

**Quality and Performance Tuning:**
```bash
python rapid.py input_pdfs/ output_dir/ \
  --text-score 0.45 \
  --images-scale 1.5 \
  --no-force-ocr \
  --normalize-output \
  --onnx-det det/inference.onnx \
  --onnx-rec rec/inference.onnx \
  --onnx-cls cls/inference.onnx \
  --rec-keys greek_keys.txt
```

**Mathematical Content Enhancement (Requires PyTorch CUDA):**
```bash
# First install PyTorch CUDA:
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1

# Then run with formula enrichment:
python rapid.py input_pdfs/ output_dir/ \
  --docling-formula \
  --formula-batch 8 \
  --docling-code \
  --onnx-det det/inference.onnx \
  --onnx-rec rec/inference.onnx \
  --onnx-cls cls/inference.onnx \
  --rec-keys greek_keys.txt
```

### Parameters Explained

- `--device cuda:0` - Use GPU acceleration for layout analysis
- `--text-score 0.45` - OCR confidence threshold (lower = more text detected)
- `--images-scale 1.5` - Scale factor for better thin glyph recognition
- `--no-force-ocr` - Use embedded PDF text when available, OCR only images
- `--normalize-output` - Apply Unicode normalization for consistent output
- `--docling-formula` - Enable mathematical formula recognition (GPU recommended)
- `--formula-batch 8` - Batch size for formula processing (adjust based on GPU memory)

### Troubleshooting

**"No class found 'rapidocr'" Error:**
- Ensure `rapidocr_onnxruntime` is installed
- Apply the Docling patch for parameter mapping
- Check that ONNX models are accessible

**GPU Provider Issues:**
- Verify CUDA installation and drivers
- Ensure only `onnxruntime-gpu` is installed (not CPU version)
- Check GPU memory availability

**Missing Dictionary Errors:**
- Generate Greek keys file from Paddle inference.yml
- Ensure the patch is applied so keys are properly passed to RapidOCR

For detailed troubleshooting, refer to the documentation included with this distribution.

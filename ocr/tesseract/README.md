# [tesseract](https://github.com/tesseract-ocr/tesseract)

## Overview

This OCR pipelne converts PDF documents to text using Tesseract OCR engine with specialized processing for Greek and English languages. The tool performs three main operations:

1. **Raw OCR Extraction** - Converts PDF pages to images and extracts text using Tesseract
2. **Text Normalization** - Strips accents and normalizes characters for consistent processing
3. **Accent Restoration** - Uses Hunspell dictionaries to restore proper Greek accents and correct spelling

## In-Depth Processing

### Preprocessing

The image preprocessing pipeline enhances OCR accuracy through several steps:

**Image Enhancement:**
- Converts PDF pages to high-resolution images (configurable DPI, default 300)
- Converts images to grayscale for better OCR performance
- Doubles the resolution using Lanczos resampling for sharper text
- Applies sharpening filter to improve character edge definition

**Character Normalization:**
- Handles common OCR misreading of Greek characters (e.g., converts micro sign µ to Greek mu μ)
- Normalizes Unicode characters to ensure consistent representation
- Prepares text for accent removal process

### Post-Processing

The post-processing stage consists of two main phases:

**Accent Stripping and Normalization:**
- Removes all Greek accents, diacritics, and combining characters using comprehensive Unicode ranges
- Handles both combining diacritical marks and precomposed accented characters
- Covers extensive Unicode ranges including:
  - Greek Extended block (U+1F00-U+1FFE) for polytonic combinations
  - All combining diacritical marks (U+0300-U+036F and extensions)
  - Greek-specific combining marks and tone markers
- Uses Unicode NFD normalization to separate base characters from diacritics

**Spell Checking and Accent Restoration:**
- Processes every word through Hunspell dictionaries for both Greek and English
- For Greek text: Uses ISO-8859-7 encoding compatibility with proper UTF-8 conversion
- Automatically detects language based on character composition
- For incorrect words: Selects the first dictionary suggestion (typically the properly accented version)
- For correct words: Preserves the original form
- Maintains formatting and spacing from the original document
- Handles encoding complexities between Hunspell's internal representation and UTF-8 output

## Output

This tool processes scanned text and produces multiple output files to help with text analysis and correction. The outputs are as follows:

1. **Raw OCR Text** (`_ocr.txt`)  
   Contains the direct output from the OCR engine

2. **Normalized Text** (`ocr_stripped.txt`)  
   A version of the text with all the accents removed

3. **Spell-Corrected Text** (`ocr_corrected.txt`)  
   The text after Hunspell's spell correction and accent restoration

4. **Unrecognized Words** (`unknown_words.txt`)  
   A list of words that could not be confidently recognized or corrected

## Installation and Setup

### Prerequisites

**System Dependencies:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ell tesseract-ocr-eng
sudo apt install python3-hunspell
sudo apt install hunspell-en-us hunspell-el
sudo apt install poppler-utils
```

**Python Dependencies:**
```bash
pip install pdf2image Pillow pytesseract tqdm hunspell
```

**Dictionary Files:**
The tool expects Hunspell dictionaries at:
- English: `/usr/share/hunspell/en_US.dic` and `/usr/share/hunspell/en_US.aff`
- Greek: `/usr/share/hunspell/el_GR.dic` and `/usr/share/hunspell/el_GR.aff`

### How to Run the OCR

**Basic Usage:**
```bash
python tesseract.py <document>.pdf
```

**With Custom DPI:**
```bash
python tesseract.py <document>.pdf <dpi>
```

**Note:** If Hunspell dictionaries are not available, the tool will still function but will only perform character normalization without accent restoration or spell checking.
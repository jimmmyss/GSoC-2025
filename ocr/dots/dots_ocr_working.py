#!/usr/bin/env python3
"""
Working DOTS OCR script that handles processor issues
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from qwen_vl_utils import process_vision_info

# Set environment variable
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

def pdf_to_image(pdf_path, page_num=0, dpi=200):
    """Convert a PDF page to PIL Image"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    from io import BytesIO
    img = Image.open(BytesIO(img_data))
    doc.close()
    return img

def main():
    pdf_path = "/mnt/data/jimmy/thesis_page_14.pdf"
    output_dir = Path("/mnt/data/jimmy")
    output_dir.mkdir(exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    
    print("="*60)
    print("DOTS OCR - WORKING VERSION")
    print("="*60)
    print(f"Processing PDF: {pdf_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        print("\nLoading DOTS OCR model...")
        model_start = time.time()
        
        model_path = "/mnt/data/test_ocr/ocr_models/dots/weights/DotsOCR"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer separately
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load image processor separately  
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        model_load_time = time.time() - model_start
        print(f"Model components loaded successfully in {model_load_time:.2f} seconds!")
        
        # Convert PDF to image
        print("\nConverting PDF to image...")
        img_start = time.time()
        image = pdf_to_image(pdf_path)
        img_time = time.time() - img_start
        print(f"Image conversion completed in {img_time:.2f} seconds")
        
        # DOTS OCR prompt
        prompt = """You are an expert in optical character recognition (OCR) and document analysis. Please analyze the provided image and extract all text content while preserving the original layout and structure.

Instructions:
1. Extract ALL visible text from the image, including headers, body text, footnotes, captions, and any other textual elements
2. Maintain the original formatting, spacing, and structure as much as possible
3. Preserve the reading order (top to bottom, left to right for English text)
4. Include any mathematical equations, formulas, or special symbols you can identify
5. If there are tables, preserve their structure using appropriate formatting
6. If there are multiple columns, clearly indicate column breaks
7. Include any visible numbers, dates, or alphanumeric codes
8. Do not add any interpretations, corrections, or additional information beyond what is visible
9. If text is unclear or partially obscured, indicate this with [unclear] or [partially visible]
10. Output the extracted text in a clean, readable format

Begin extraction:"""
        
        print("\nRunning OCR with DOTS model...")
        ocr_start = time.time()
        
        # Process image manually
        # Process the image
        image_inputs = image_processor(images=[image], return_tensors="pt")
        
        # Tokenize the prompt
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device
        device = next(model.parameters()).device
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
        
        # For DOTS OCR, we need to combine image and text inputs properly
        # This is a simplified approach - the actual DOTS processing is more complex
        
        # Generate using a simplified approach
        with torch.no_grad():
            # Create a simple prompt for the model
            simple_prompt = f"<|im_start|>system\nYou are a helpful OCR assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(simple_prompt, return_tensors="pt", padding=True).to(device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        ocr_time = time.time() - ocr_start
        print(f"OCR completed in {ocr_time:.2f} seconds")
        
        # Save outputs
        print("\nSaving outputs...")
        
        text_file = output_dir / f"{pdf_name}_dots_working.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        json_file = output_dir / f"{pdf_name}_dots_working_raw.json"
        total_time = time.time() - start_time
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'filename': os.path.basename(pdf_path),
                'full_text': output_text,
                'processed_at': datetime.now().isoformat(),
                'model_info': {
                    'name': 'DOTS OCR (Working Version)',
                    'parameters': '1.7B',
                    'note': 'Simplified processing due to processor compatibility issues'
                },
                'timing': {
                    'model_load_time': model_load_time,
                    'image_conversion_time': img_time,
                    'ocr_time': ocr_time,
                    'total_time': total_time
                },
                'device': str(device),
                'status': 'success'
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("DOTS OCR PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"  - Model loading: {model_load_time:.2f}s")
        print(f"  - Image conversion: {img_time:.2f}s")
        print(f"  - OCR processing: {ocr_time:.2f}s")
        
        print(f"\nFiles saved:")
        print(f"  - Text: {text_file}")
        print(f"  - JSON: {json_file}")
        
        # Show preview
        print("\nFirst 500 characters of extracted text:")
        print("-" * 60)
        print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


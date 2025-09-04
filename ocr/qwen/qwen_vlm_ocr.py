#!/usr/bin/env python3
"""
OCR pipeline using Qwen2.5-VL-3B-Instruct vision-language model
This model can read and understand text in images, including Greek text
"""

import os
import sys
import time
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
import re

def setup_qwen_model():
    """Setup Qwen2.5-VL-3B-Instruct model"""
    print("Loading Qwen2.5-VL-3B-Instruct model...")
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Check device
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    
    return model, processor

def extract_text_from_page(image, model, processor):
    """Extract text from a single page using Qwen VL model"""
    
    # Create conversation for text extraction
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # PIL Image object
                },
                {
                    "type": "text", 
                    "text": "Please extract and transcribe ALL text from this image exactly as it appears. Include Greek text, English text, numbers, and maintain the original formatting as much as possible. Do not translate or interpret, just transcribe what you see."
                }
            ],
        }
    ]
    
    # Prepare inputs using the proper Qwen VL processing
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.1,
            top_p=0.95,
        )
    
    # Decode only the generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def process_pdf_with_qwen(pdf_path, dpi=300, save_output=True, output_path=None):
    """Process PDF using Qwen2.5-VL model"""
    
    print(f"Processing: {pdf_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup model
    setup_start = time.time()
    try:
        model, processor = setup_qwen_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have enough GPU memory and the required packages installed")
        return None
    
    setup_time = time.time() - setup_start
    print(f"Model setup time: {setup_time:.2f} seconds")
    
    # Phase 1: Convert PDF to images
    phase1_start = time.time()
    print("\nPhase 1: Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=dpi)
    phase1_time = time.time() - phase1_start
    print(f"  - Completed in {phase1_time:.2f} seconds for {len(images)} pages")
    print(f"  - Average: {phase1_time/len(images):.2f} seconds per page")
    
    # Phase 2: Extract text with Qwen VL
    phase2_start = time.time()
    print("\nPhase 2: Extracting text with Qwen2.5-VL...")
    all_text = []
    extraction_times = []
    
    for i, img in enumerate(tqdm(images, desc="Processing pages")):
        page_start = time.time()
        
        try:
            # Extract text from page
            text = extract_text_from_page(img, model, processor)
            
            # Clean up the output (remove any instruction artifacts)
            text = text.strip()
            
            all_text.append(f"\n--- Page {i + 1} ---\n{text}")
            
        except Exception as e:
            print(f"\nError processing page {i + 1}: {e}")
            all_text.append(f"\n--- Page {i + 1} ---\n[Error: Could not process this page]")
        
        page_time = time.time() - page_start
        extraction_times.append(page_time)
        
        # Print progress every 5 pages
        if (i + 1) % 5 == 0:
            avg_so_far = sum(extraction_times) / len(extraction_times)
            print(f"\nProcessed {i + 1}/{len(images)} pages. Average time: {avg_so_far:.2f}s/page")
    
    full_text = "\n".join(all_text)
    phase2_time = time.time() - phase2_start
    avg_extraction_time = sum(extraction_times) / len(extraction_times) if extraction_times else 0
    
    print(f"\n  - Completed in {phase2_time:.2f} seconds")
    print(f"  - Average: {avg_extraction_time:.2f} seconds per page")
    print(f"  - Min/Max per page: {min(extraction_times):.2f}s / {max(extraction_times):.2f}s")
    
    # Phase 3: Save output
    if save_output:
        phase3_start = time.time()
        print("\nPhase 3: Saving output...")
        
        if output_path:
            output_file = output_path
        else:
            # Default to ocr_test directory
            pdf_basename = os.path.basename(pdf_path)
            pdf_name = os.path.splitext(pdf_basename)[0]
            output_dir = "/mnt/data/test_ocr/ocr_test"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{pdf_name}_qwen_vlm.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        phase3_time = time.time() - phase3_start
        print(f"  - Output saved to: {output_file}")
        print(f"  - Save time: {phase3_time:.2f} seconds")
    
    # Summary
    total_time = time.time() - setup_start
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Total pages processed: {len(images)}")
    print(f"Average time per page: {total_time/len(images):.2f} seconds")
    print(f"\nPhase breakdown:")
    print(f"  Model setup: {setup_time:.2f}s ({setup_time/total_time*100:.1f}%)")
    print(f"  PDF to images: {phase1_time:.2f}s ({phase1_time/total_time*100:.1f}%)")
    print(f"  Text extraction: {phase2_time:.2f}s ({phase2_time/total_time*100:.1f}%)")
    if save_output:
        print(f"  Save output: {phase3_time:.2f}s ({phase3_time/total_time*100:.1f}%)")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return full_text

def main():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL based OCR for PDFs')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PDF conversion (default: 300)')
    parser.add_argument('--output', help='Output file path (default: input_qwen_vlm.txt)')
    
    args = parser.parse_args()
    
    # Process the PDF
    process_pdf_with_qwen(args.pdf_file, dpi=args.dpi, output_path=args.output)

if __name__ == "__main__":
    main()
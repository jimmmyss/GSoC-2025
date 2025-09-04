#!/usr/bin/env python3
"""
Nanonets OCR using Hugging Face model for PDF processing
Uses the nanonets/Nanonets-OCR-s model from Hugging Face
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import torch
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


class NanonetsHFOCR:
    """Nanonets OCR using Hugging Face model for processing PDFs"""
    
    def __init__(self, model_path: str = "nanonets/Nanonets-OCR-s", device: Optional[str] = None):
        """
        Initialize Nanonets Hugging Face OCR model
        
        Args:
            model_path: Path to the model on Hugging Face
            device: Device to run on (cuda, cpu, or auto)
        """
        print(f"Loading model: {model_path}")
        print(f"Note: This model is approximately 7.5GB in size. First time loading will download it.")
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model with appropriate settings
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,  # Required for custom model code
        }
        
        # Load model, tokenizer, and processor with progress indication
        print("Loading model components...")
        with tqdm(total=3, desc="Loading", unit="component") as pbar:
            # Load model
            pbar.set_description("Loading model (7.5GB)")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs
            )
            pbar.update(1)
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            
            # Load tokenizer
            pbar.set_description("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            pbar.update(1)
            
            # Load processor
            pbar.set_description("Loading processor")
            self.processor = AutoProcessor.from_pretrained(model_path)
            pbar.update(1)
        
        print("Model loaded successfully")
    
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF pages to PIL images
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL images
        """
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Render at 300 DPI for good quality
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            import io
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
        return images
    
    def process_image(self, image: Image.Image) -> str:
        """
        Process a single image with the OCR model
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text in markdown format
        """
        # Prepare the prompt for OCR
        prompt = """Extract the text from the above document as if you were reading it naturally.
Include all text content. 
For any equations or mathematical expressions, output them in LaTeX format.
For any tables present, output them in HTML table format. 
For any images, describe them using <img> tags with descriptive alt text.
For any signatures found on the document, output them in a <signature> tag.
Ensure that the formatting of headers, lists, and code blocks is done with markdown. 
Do NOT add extra information or make assumptions. ONLY transcribe what you see on the page."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process the input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.1,
                do_sample=False,
            )
        
        # Decode the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file using Nanonets HF OCR
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            OCR results dictionary
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        images = self.pdf_to_images(pdf_path)
        print(f"Found {len(images)} pages")
        
        # Process each page
        pages_data = []
        total_text = []
        
        print("Processing pages...")
        for i, image in enumerate(tqdm(images, desc="Pages", unit="page")):
            start_time = time.time()
            
            page_text = self.process_image(image)
            process_time = time.time() - start_time
            
            pages_data.append({
                "page_number": i + 1,
                "text": page_text,
                "process_time": process_time
            })
            total_text.append(f"## Page {i+1}\n\n{page_text}")
            
            tqdm.write(f"  Page {i+1} completed in {process_time:.2f} seconds")
        
        # Create result dictionary
        result = {
            "filename": os.path.basename(pdf_path),
            "total_pages": len(images),
            "pages": pages_data,
            "full_text": "\n\n---\n\n".join(total_text),
            "processed_at": datetime.now().isoformat()
        }
        
        return result
    
    def save_results(self, pdf_path: str, ocr_result: Dict, output_dir: str):
        """
        Save OCR results to files
        
        Args:
            pdf_path: Original PDF path
            ocr_result: OCR result dictionary
            output_dir: Directory to save results
        """
        pdf_name = Path(pdf_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw JSON response
        json_path = output_path / f"{pdf_name}_nanonets_raw.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_result, f, indent=2, ensure_ascii=False)
        
        # Save full text
        text_path = output_path / f"{pdf_name}_nanonets.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(ocr_result['full_text'])
        
        # Save markdown version
        md_path = output_path / f"{pdf_name}_nanonets.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Nanonets OCR Output: {pdf_name}\n\n")
            f.write(f"Processed on: {ocr_result['processed_at']}\n")
            f.write(f"Total pages: {ocr_result['total_pages']}\n\n")
            f.write("---\n\n")
            f.write(ocr_result['full_text'])
        
        print(f"\nResults saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - Text: {text_path}")
        print(f"  - Markdown: {md_path}")
        
        return md_path


def main():
    """Main function to run Nanonets HF OCR on PDF files"""
    parser = argparse.ArgumentParser(description='Run Nanonets HF OCR on PDF files')
    parser.add_argument('pdf_files', nargs='+', help='PDF files to process')
    parser.add_argument('--output-dir', default='./nanonets_output', help='Output directory')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], help='Device to use')
    parser.add_argument('--model-path', default='nanonets/Nanonets-OCR-s', help='Model path on Hugging Face')
    
    args = parser.parse_args()
    
    # Initialize Nanonets HF client
    print("Initializing Nanonets HF OCR...")
    client = NanonetsHFOCR(model_path=args.model_path, device=args.device)
    
    # Process each PDF
    for pdf_path in args.pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print('='*60)
        
        try:
            # Process PDF
            start_time = time.time()
            result = client.process_pdf(pdf_path)
            total_time = time.time() - start_time
            
            print(f"\nTotal OCR time: {total_time:.2f} seconds")
            
            # Save results
            client.save_results(pdf_path, result, args.output_dir)
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
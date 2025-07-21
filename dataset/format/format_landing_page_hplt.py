#!/usr/bin/env python3
"""
JSONL to Parquet Batch Converter
================================

This script processes all JSONL files in a specified folder in numerical order,
converts them to a single parquet file using efficient batching and memory management.

Usage: python extract.py <folder_path>

Features:
- Processes JSONL files in numerical order (1.jsonl, 2.jsonl, etc.)
- Batches of 100,000 records to avoid memory issues
- Proper memory management with garbage collection
- Progress bars for files and GB remaining
- Adds character count field from text content
- Preserves all original fields from JSONL files
- Maintains exact field structure as existing parquet files

"""

import pandas as pd
import json
import sys
import os
import argparse
import gc
import gzip
import psutil
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    return os.path.getsize(file_path) / (1024 ** 3)

def get_jsonl_files_in_order(folder_path: str) -> List[str]:
    """
    Get all JSONL files in the folder sorted by numerical order.
    
    Args:
        folder_path (str): Path to folder containing JSONL files
        
    Returns:
        List[str]: Sorted list of JSONL file paths
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all .jsonl and .jsonl.gz files
    jsonl_files = []
    for pattern in ['*.jsonl', '*.jsonl.gz']:
        jsonl_files.extend(folder.glob(pattern))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {folder_path}")
    
    # Sort by numerical order (extract number from filename)
    def extract_number(filepath):
        try:
            # Extract number from filename (e.g., "1.jsonl" -> 1)
            basename = filepath.stem
            if basename.endswith('.jsonl'):
                basename = basename[:-6]  # Remove .jsonl extension
            return int(basename)
        except ValueError:
            # If no number found, use filename for sorting
            return float('inf')
    
    jsonl_files.sort(key=extract_number)
    
    print(f"📁 Found {len(jsonl_files)} JSONL files:")
    for i, file in enumerate(jsonl_files, 1):
        size_gb = get_file_size_gb(file)
        print(f"   {i}. {file.name} ({size_gb:.2f} GB)")
    
    return [str(f) for f in jsonl_files]

def is_landing_page_path(url):
    """
    Check if URL has a landing page path (with or without trailing slash)
    Improved version that handles trailing slashes correctly
    """
    # Define the landing page patterns (without trailing slash)
    landing_files = [
        'home.php', 'main.html', 'main.php', 'default.html', 'default.aspx',
        'landing.html', 'landing.php', 'welcome.html', 'welcome.php',
        'start.html', 'start.php', 'root.html', 'app.html', 'home.aspx',
        'portal.html', 'index.html', 'index.php'
    ]
    
    landing_paths = [
        'home', 'main', 'landing', 'start', 'welcome', 
        'dashboard', 'intro', 'index'
    ]
    
    try:
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment
        
        # No query parameters or fragments allowed
        if query or fragment:
            return False
        
        # Remove trailing slash for comparison
        clean_path = path.rstrip('/')
        
        # Also handle case where path doesn't start with /
        if not clean_path.startswith('/'):
            clean_path = '/' + clean_path
        
        # Check if path exactly matches landing files (with leading slash)
        for landing_file in landing_files:
            if clean_path == f'/{landing_file}':
                return True
        
        # Check if path exactly matches landing paths (with leading slash)
        for landing_path in landing_paths:
            if clean_path == f'/{landing_path}':
                return True
                
        return False
    except:
        return False

def count_text_characters(entry: Dict[str, Any]) -> int:
    """
    Count total characters in text-related fields.
    Based on the existing parquet file structure, mainly focuses on 'text' field.
    
    Args:
        entry (Dict[str, Any]): JSONL entry
        
    Returns:
        int: Total character count
    """
    # Primary text field (as seen in existing parquet structure)
    total_chars = 0
    
    # Main text field - this is the primary field in the existing structure
    if 'text' in entry and entry['text'] is not None:
        total_chars += len(str(entry['text']))
    
    # Additional text fields that might be present
    additional_text_fields = ['content', 'body', 'description', 'title']
    for field in additional_text_fields:
        if field in entry and entry[field] is not None:
            total_chars += len(str(entry[field]))
    
    return total_chars

def process_jsonl_file(file_path: str, output_file: str, batch_size: int = 100000, 
                      is_first_file: bool = True) -> int:
    """
    Process a single JSONL file and append to parquet file.
    
    Args:
        file_path (str): Path to JSONL file
        output_file (str): Path to output parquet file
        batch_size (int): Number of records per batch
        is_first_file (bool): Whether this is the first file being processed
        
    Returns:
        int: Number of records processed
    """
    print(f"\n📄 Processing: {os.path.basename(file_path)}")
    
    # Get file size for progress tracking
    file_size_gb = get_file_size_gb(file_path)
    print(f"   File size: {file_size_gb:.2f} GB")
    
    # Determine if file is gzipped
    is_gzipped = file_path.endswith('.gz')
    file_opener = gzip.open if is_gzipped else open
    file_mode = 'rt' if is_gzipped else 'r'
    
    batch_data = []
    total_processed = 0
    bytes_processed = 0
    batch_count = 0
    
    # Progress bar for GB remaining
    with tqdm(total=file_size_gb, desc="Processing GB", unit="GB", position=1) as gb_pbar:
        with file_opener(file_path, file_mode, encoding='utf-8') as f:
            for line in f:
                try:
                    # Parse JSON line
                    entry = json.loads(line.strip())
                    
                    # Add character count field as the last field
                    char_count = count_text_characters(entry)
                    entry['text_char_count'] = char_count
                    
                    batch_data.append(entry)
                    total_processed += 1
                    
                    # Update bytes processed for progress bar
                    bytes_processed += len(line.encode('utf-8'))
                    gb_processed = bytes_processed / (1024 ** 3)
                    gb_pbar.update(gb_processed - gb_pbar.n)
                    
                    # Process batch when full
                    if len(batch_data) >= batch_size:
                        batch_count += 1
                        save_batch_to_parquet(batch_data, output_file, 
                                            is_first_file and batch_count == 1)
                        
                        # Clear batch data and collect garbage
                        batch_data = []
                        gc.collect()
                        
                        # Show memory usage
                        memory_mb = psutil.Process().memory_info().rss / (1024 ** 2)
                        tqdm.write(f"   Batch {batch_count} saved. Memory: {memory_mb:.1f} MB")
                
                except json.JSONDecodeError as e:
                    tqdm.write(f"   ⚠️  JSON decode error: {e}")
                    continue
                except Exception as e:
                    tqdm.write(f"   ⚠️  Error processing line: {e}")
                    continue
    
    # Save remaining data
    if batch_data:
        batch_count += 1
        save_batch_to_parquet(batch_data, output_file, 
                            is_first_file and batch_count == 1)
        batch_data = []
        gc.collect()
    
    print(f"   ✅ Processed {total_processed:,} records in {batch_count} batches")
    return total_processed

def save_batch_to_parquet(data: List[Dict[str, Any]], output_file: str, 
                         is_first_batch: bool) -> None:
    """
    Save a batch of data to parquet file.
    Maintains the exact column order from the existing parquet structure.
    
    Args:
        data (List[Dict[str, Any]]): List of records
        output_file (str): Path to output parquet file
        is_first_batch (bool): Whether this is the first batch
    """
    if not data:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Define the expected column order based on the existing parquet structure
    expected_columns = [
        'f', 'o', 's', 'rs', 'u', 'c', 'ts', 'collection', 'lang', 'prob', 
        'text', 'seg_langs', 'robotstxt', 'id', 'filter', 'pii', 'doc_scores'
    ]
    
    # Get all columns from the DataFrame
    all_columns = []
    
    # Add expected columns first (if they exist)
    for col in expected_columns:
        if col in df.columns:
            all_columns.append(col)
    
    # Add any additional columns that aren't in the expected list
    for col in df.columns:
        if col not in expected_columns and col != 'text_char_count':
            all_columns.append(col)
    
    # Add text_char_count as the last column
    if 'text_char_count' in df.columns:
        all_columns.append('text_char_count')
    
    # Reorder DataFrame columns
    df = df.reindex(columns=all_columns, fill_value=None)
    
    # Save to parquet
    if is_first_batch:
        # Create new file
        df.to_parquet(output_file, index=False, engine='pyarrow')
    else:
        # Append to existing file
        # Read existing schema to ensure compatibility
        existing_df = pd.read_parquet(output_file)
        
        # Align columns with existing file
        existing_columns = list(existing_df.columns)
        
        # Combine columns (existing first, then new ones, text_char_count last)
        final_columns = []
        for col in existing_columns:
            if col != 'text_char_count':
                final_columns.append(col)
        
        # Add new columns (if any)
        for col in all_columns:
            if col not in final_columns and col != 'text_char_count':
                final_columns.append(col)
        
        # Add text_char_count as last column
        if 'text_char_count' in existing_columns or 'text_char_count' in all_columns:
            final_columns.append('text_char_count')
        
        # Reindex both DataFrames
        existing_df = existing_df.reindex(columns=final_columns, fill_value=None)
        df = df.reindex(columns=final_columns, fill_value=None)
        
        # Combine and save
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(output_file, index=False, engine='pyarrow')

def analyze_output_parquet(output_file: str) -> None:
    """
    Analyze the final parquet file and show statistics.
    
    Args:
        output_file (str): Path to output parquet file
    """
    if not os.path.exists(output_file):
        print(f"⚠️  Output file not found: {output_file}")
        return
    
    try:
        # Read parquet file metadata
        df = pd.read_parquet(output_file)
        
        print(f"\n📊 Output File Analysis:")
        print(f"   File: {output_file}")
        print(f"   Total records: {len(df):,}")
        print(f"   File size: {get_file_size_gb(output_file):.2f} GB")
        print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
        
        # Analyze text character count distribution
        if 'text_char_count' in df.columns:
            char_stats = df['text_char_count'].describe()
            print(f"\n📝 Text Character Count Statistics:")
            print(f"   Mean: {char_stats['mean']:.0f}")
            print(f"   Median: {char_stats['50%']:.0f}")
            print(f"   Min: {char_stats['min']:.0f}")
            print(f"   Max: {char_stats['max']:.0f}")
            print(f"   Records with text: {(df['text_char_count'] > 0).sum():,}")
            print(f"   Records without text: {(df['text_char_count'] == 0).sum():,}")
        
        # Show sample of data
        print(f"\n📋 Sample Records:")
        for i, (idx, row) in enumerate(df.head(3).iterrows()):
            print(f"   Record {i+1}:")
            for col in df.columns:
                value = row[col]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"     {col}: {value}")
            print()
    
    except Exception as e:
        print(f"⚠️  Error analyzing output file: {e}")

def main():
    """Main function to process JSONL files and create parquet output."""
    parser = argparse.ArgumentParser(
        description='Convert JSONL files in a folder to a single parquet file'
    )
    parser.add_argument('folder', help='Path to folder containing JSONL files')
    parser.add_argument('--output', '-o', default='combined_data.parquet',
                       help='Output parquet file name (default: combined_data.parquet)')
    parser.add_argument('--batch-size', '-b', type=int, default=100000,
                       help='Batch size for processing (default: 100000)')
    
    args = parser.parse_args()
    
    print("📦 JSONL to Parquet Batch Converter")
    print("=" * 50)
    print(f"📁 Input folder: {args.folder}")
    print(f"📄 Output file: {args.output}")
    print(f"📊 Batch size: {args.batch_size:,}")
    print("=" * 50)
    
    try:
        # Get JSONL files in order
        jsonl_files = get_jsonl_files_in_order(args.folder)
        
        # Calculate total size
        total_size_gb = sum(get_file_size_gb(f) for f in jsonl_files)
        print(f"\n📊 Total data size: {total_size_gb:.2f} GB")
        
        # Process files with overall progress bar
        total_processed = 0
        start_time = time.time()
        
        with tqdm(total=len(jsonl_files), desc="Processing files", 
                 unit="file", position=0) as file_pbar:
            
            for i, file_path in enumerate(jsonl_files):
                file_pbar.set_description(f"Processing file {i+1}/{len(jsonl_files)}")
                
                # Process file
                records_processed = process_jsonl_file(
                    file_path, 
                    args.output, 
                    args.batch_size,
                    is_first_file=(i == 0)
                )
                
                total_processed += records_processed
                file_pbar.update(1)
                
                # Show progress
                elapsed = time.time() - start_time
                avg_speed = total_processed / elapsed if elapsed > 0 else 0
                file_pbar.set_postfix({
                    'Records': f"{total_processed:,}",
                    'Speed': f"{avg_speed:.0f}/sec"
                })
        
        # Final statistics
        elapsed = time.time() - start_time
        print(f"\n🎉 Processing completed!")
        print(f"   Total records processed: {total_processed:,}")
        print(f"   Total time: {elapsed:.2f} seconds")
        print(f"   Average speed: {total_processed / elapsed:.0f} records/sec")
        print(f"   Output file: {args.output}")
        
        # Analyze output
        analyze_output_parquet(args.output)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
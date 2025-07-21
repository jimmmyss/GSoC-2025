#!/usr/bin/env python3

import pandas as pd
import json
from urllib.parse import urlparse
from collections import Counter
import glob
import os
from tqdm import tqdm
import sys
import gc
import time

def extract_domain_from_url(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain:
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        return None
    except:
        return None

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_checkpoint_path(file_path):
    """Get checkpoint file path for a given input file."""
    base_name = os.path.basename(file_path)
    return os.path.join('oscar_domains.parquet', 'checkpoints', f'{base_name}.checkpoint')

def get_progress_file():
    """Get path to the progress tracking file."""
    return os.path.join('oscar_domains.parquet', 'progress.json')

def save_progress(completed_files):
    """Save list of completed files to progress tracker."""
    progress_file = get_progress_file()
    with open(progress_file, 'w') as f:
        json.dump({'completed_files': completed_files}, f)

def load_progress():
    """Load list of completed files from progress tracker."""
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed_files', []))
    return set()

def load_checkpoint(file_path):
    """Load checkpoint for a file if it exists."""
    checkpoint_path = get_checkpoint_path(file_path)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(file_path, processed_lines):
    """Save checkpoint for a file."""
    checkpoint_path = get_checkpoint_path(file_path)
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'processed_lines': processed_lines
        }, f)

def count_lines_efficiently(file_path):
    """Count lines in a file without loading it entirely into memory."""
    def blocks(file, size=65536):
        while True:
            b = file.read(size)
            if not b: break
            yield b
    
    # Get file size for progress bar
    file_size = os.path.getsize(file_path)
    
    with tqdm(total=file_size, desc="Counting lines", unit='B', unit_scale=True) as pbar:
        with open(file_path, "rb") as f:
            total_lines = 0
            for block in blocks(f):
                total_lines += block.count(b"\n")
                pbar.update(len(block))
    
    return total_lines

def process_oscar_file(file_path, position=0):
    """Process a single OSCAR file and extract domains."""
    file_base_name = os.path.basename(file_path)
    print(f"\nProcessing {file_base_name}")
    
    # Initialize counters and data structures
    domain_counts = {}
    processed_lines = 0
    
    # Check for checkpoint
    checkpoint = load_checkpoint(file_path)
    if checkpoint:
        processed_lines = checkpoint['processed_lines']
        print(f"Resuming from line {processed_lines:,}")
    
    try:
        # Count total lines efficiently with progress bar
        print("Counting total lines...")
        total_lines = count_lines_efficiently(file_path)
        print(f"Found {total_lines:,} lines to process")
        
        # Create progress bar for this file
        with tqdm(total=total_lines, initial=processed_lines,
                 desc=f"Processing {file_base_name}", position=position) as pbar:
            
            # Process file in chunks to avoid memory issues
            with open(file_path, 'r') as f:
                # Skip to last processed line if resuming
                if processed_lines > 0:
                    print(f"Skipping to line {processed_lines:,}...")
                    for _ in range(processed_lines):
                        next(f)
                
                # Process remaining lines
                last_gc_time = time.time()
                
                for line in f:
                    try:
                        data = json.loads(line)
                        url = data.get('warc_headers', {}).get('warc-target-uri')
                        if url:
                            domain = extract_domain_from_url(url)
                            if domain:
                                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                        
                        processed_lines += 1
                        
                        # Save checkpoint periodically
                        if processed_lines % 100000 == 0:
                            save_checkpoint(file_path, processed_lines)
                            print(f"\nProcessed {processed_lines:,} lines, found {len(domain_counts):,} unique domains")
                        
                        # More frequent garbage collection based on time
                        current_time = time.time()
                        if current_time - last_gc_time > 30:  # GC every 30 seconds
                            gc.collect()
                            last_gc_time = current_time
                        
                        pbar.update(1)
                        
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
                    except Exception as e:
                        print(f"\nError processing line: {e}")
                        continue
    
    except Exception as e:
        print(f"\nError processing file {file_path}: {e}")
        return None
    
    # Convert final results to DataFrame
    print(f"\nFile processing complete. Converting {len(domain_counts):,} domains to DataFrame...")
    
    try:
        # Convert to DataFrame in chunks to avoid memory spikes
        chunk_size = 100000  # Process 100k domains at a time
        df_chunks = []
        items = list(domain_counts.items())
        
        # Clear domain_counts to free memory
        domain_counts.clear()
        gc.collect()
        
        # Process items in chunks with progress bar
        with tqdm(total=len(items), desc="Creating DataFrame", position=position) as pbar:
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                df_chunk = pd.DataFrame(chunk, columns=['domain', 'count'])
                df_chunks.append(df_chunk)
                del chunk
                pbar.update(chunk_size)
        
        print("\nMerging chunks...")
        # Combine chunks
        df = pd.concat(df_chunks, ignore_index=True)
        del df_chunks
        gc.collect()
        
        print("Sorting results...")
        # Sort by count in descending order
        df.sort_values('count', ascending=False, inplace=True)
        
        # Save result
        output_file = os.path.join('oscar_domains.parquet', 'merged', f'{file_base_name}_domains.parquet')
        print(f"Saving results to {output_file}")
        df.to_parquet(output_file, engine='pyarrow', index=False)
        
        # Print summary
        print(f"\nFile summary for {file_base_name}:")
        print(f"- Total lines processed: {processed_lines:,}")
        print(f"- Unique domains found: {len(df):,}")
        print(f"- Total domain occurrences: {df['count'].sum():,}")
        
        return output_file
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None

def create_oscar_parquet(input_dir):
    """Create parquet file with domain statistics from OSCAR dataset."""
    
    # Create working directory
    temp_dir = 'oscar_domains'
    ensure_dir(temp_dir)
    ensure_dir(os.path.join(temp_dir, 'merged'))
    ensure_dir(os.path.join(temp_dir, 'checkpoints'))
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Looking for OSCAR files in: {input_dir}")
    
    # Find all OSCAR jsonl files
    files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    
    if not files:
        print("❌ No OSCAR files found!")
        print("Make sure the path contains .jsonl files")
        return
    
    # Sort files to process them in order
    files.sort()
    
    # Load progress from previous run if exists
    completed_files = load_progress()
    remaining_files = [f for f in files if f not in completed_files]
    
    if completed_files:
        print(f"\nFound previous progress:")
        print(f"- Completed files: {len(completed_files)}")
        print(f"- Remaining files: {len(remaining_files)}")
        
        if not remaining_files:
            print("\nAll files have been processed!")
            print("If you want to reprocess, delete the oscar_domains directory first.")
            return
    
    print(f"\nProcessing {len(remaining_files)} files...")
    processed_files = []
    
    # Create progress bar for overall file processing
    with tqdm(total=len(remaining_files), desc="Overall progress", position=0) as main_pbar:
        # Process each remaining file
        for idx, file in enumerate(remaining_files):
            print(f"\nProcessing file {idx + 1} of {len(remaining_files)}: {os.path.basename(file)}")
            output_file = process_oscar_file(file, position=1)
            
            if output_file:
                processed_files.append(output_file)
                completed_files.add(file)
                save_progress(list(completed_files))  # Save progress after each file
            
            main_pbar.update(1)
            
            # Clear the progress bar lines
            sys.stdout.write("\033[K")
            sys.stdout.write("\033[K")
    
    if not processed_files:
        print("\n❌ No data was processed!")
        # Clean up working directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    # Final merge of all processed files
    print("\nMerging all processed files into final result...")
    final_df = pd.DataFrame(columns=['domain', 'count'])
    
    # Process files in chunks
    chunk_size = 5  # Process 5 files at a time
    for i in range(0, len(processed_files), chunk_size):
        chunk_files = processed_files[i:i + chunk_size]
        print(f"\nMerging files {i+1}-{min(i+chunk_size, len(processed_files))} of {len(processed_files)}...")
        
        # Read and merge chunk of files
        chunk_dfs = []
        for file in chunk_files:
            try:
                df = pd.read_parquet(file)
                chunk_dfs.append(df)
                del df
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
        if chunk_dfs:
            # Merge chunk results
            chunk_result = pd.concat(chunk_dfs, ignore_index=True)
            chunk_result = chunk_result.groupby('domain', as_index=False)['count'].sum()
            
            # Merge with final results
            if final_df.empty:
                final_df = chunk_result
            else:
                final_df = pd.concat([final_df, chunk_result], ignore_index=True)
                final_df = final_df.groupby('domain', as_index=False)['count'].sum()
            
            # Clear chunk memory
            del chunk_dfs
            del chunk_result
            gc.collect()
    
    # Final sort
    print("\nSorting final results...")
    final_df.sort_values('count', ascending=False, inplace=True)
    
    # Save final result in the same directory as this script
    output_file = os.path.join(script_dir, 'oscar_domains.parquet')
    print(f"Saving final results to {output_file}")
    final_df.to_parquet(output_file, engine='pyarrow', index=False)
    
    # Clean up working directory
    print("\nCleaning up working directory...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✅ OSCAR domain statistics created successfully!")
    print(f"Total unique domains: {len(final_df):,}")
    print(f"Total domain occurrences: {final_df['count'].sum():,}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create domain statistics from OSCAR dataset')
    parser.add_argument('input_dir', help='Directory containing OSCAR JSON files')
    args = parser.parse_args()
    
    create_oscar_parquet(args.input_dir) 

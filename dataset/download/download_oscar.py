#!/usr/bin/env python3

import os
import sys
from huggingface_hub import hf_hub_download
import concurrent.futures
from tqdm import tqdm

def download_file(filename, token):
    """Download a single file from the dataset"""
    try:
        print(f"Starting download: {filename}")
        local_filename = hf_hub_download(
            repo_id="oscar-corpus/OSCAR-2301",
            filename=f"oscar/{filename}",
            repo_type="dataset",
            local_dir=".",
            token=token
        )
        print(f"Completed download: {filename}")
        return filename, True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return filename, False

def main():
    # Check if token is provided as argument
    if len(sys.argv) != 2:
        print("Usage: python download_oscar.py <huggingface_token>")
        print("Please provide your Hugging Face API token as an argument.")
        sys.exit(1)
    
    token = sys.argv[1]
    
    # Create oscar directory if it doesn't exist
    os.makedirs("oscar", exist_ok=True)
    
    # Generate list of all oscar files (including parts 54-60)
    files_to_download = []
    
    # Add checksum file
    files_to_download.append("checksum.sha256")
    
    # Add all el_meta_part files (from 1 to 60 to include the missing files)
    for i in range(1, 61):  # 1 to 60 inclusive
        files_to_download.append(f"el_meta_part_{i}.jsonl.zst")
    
    print(f"Preparing to download {len(files_to_download)} files ...")
    
    # Download files with concurrent processing (but not too many at once to avoid overwhelming the server)
    max_workers = 3  # Conservative number to avoid rate limiting
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_filename = {executor.submit(download_file, filename, token): filename 
                             for filename in files_to_download}
        
        # Process completed downloads
        completed = 0
        failed = 0
        
        for future in concurrent.futures.as_completed(future_to_filename):
            filename, success = future.result()
            if success:
                completed += 1
            else:
                failed += 1
            
            print(f"Progress: {completed + failed}/{len(files_to_download)} "
                  f"(Success: {completed}, Failed: {failed})")
    
    print(f"\nDownload completed!")
    print(f"Successfully downloaded: {completed} files")
    print(f"Failed downloads: {failed} files")

if __name__ == "__main__":
    main() 
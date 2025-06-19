#!/usr/bin/env python3

import os
import requests
import concurrent.futures
from tqdm import tqdm
from urllib.parse import urlparse
import threading
import time

def download_file(url):
    """Download a single file from the given URL"""
    try:
        # Extract filename from URL
        filename = os.path.basename(urlparse(url).path)
        local_path = os.path.join("hplt", filename)
        
        print(f"Starting download: {filename}")
        
        # Download with streaming to handle large files
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            print(f"File {filename} size: {total_size / (1024*1024*1024):.2f} GB")
            
            with open(local_path, 'wb') as f:
                if total_size == 0:
                    # No content-length header
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                else:
                    # Use tqdm for progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, position=int(filename[0])-1) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"Completed download: {filename}")
        return filename, True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return os.path.basename(urlparse(url).path), False

def main():
    # Create hplt directory if it doesn't exist
    os.makedirs("hplt", exist_ok=True)
    
    # List of HPLT Greek files to download
    urls_to_download = [
        "https://data.hplt-project.org/two/cleaned/ell_Grek/1.jsonl.zst",
        "https://data.hplt-project.org/two/cleaned/ell_Grek/2.jsonl.zst",
        "https://data.hplt-project.org/two/cleaned/ell_Grek/3.jsonl.zst",
        "https://data.hplt-project.org/two/cleaned/ell_Grek/4.jsonl.zst"
    ]
    
    print(f"Preparing to download {len(urls_to_download)} HPLT Greek files...")
    print("All files will be downloaded concurrently...")
    
    # Download all files concurrently - set to 4 to download all at once
    max_workers = 4  # Download all 4 files simultaneously
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {executor.submit(download_file, url): url 
                        for url in urls_to_download}
        
        # Process completed downloads
        completed = 0
        failed = 0
        
        for future in concurrent.futures.as_completed(future_to_url):
            filename, success = future.result()
            if success:
                completed += 1
            else:
                failed += 1
            
            print(f"Progress: {completed + failed}/{len(urls_to_download)} "
                  f"(Success: {completed}, Failed: {failed})")
    
    print(f"\nHPLT Greek files download completed!")
    print(f"Successfully downloaded: {completed} files")
    print(f"Failed downloads: {failed} files")

if __name__ == "__main__":
    main() 
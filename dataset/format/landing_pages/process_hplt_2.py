#!/usr/bin/env python3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import re
from tqdm import tqdm
import os

def merge_hplt_2_streaming():
    """
    Merge hplt_2 batch files by streaming each batch directly to the output parquet file.
    This approach never loads more than one batch into memory at a time.
    """
    batch_dir = Path("hplt_deduped/hplt_2")
    output_file = Path("hplt_deduped/hplt_cleaned_metadata_2.parquet")
    
    print("=== STREAMING MERGE OF HPLT_2 BATCH FILES ===")
    
    # Get all batch files and sort them
    batch_files = []
    for file in batch_dir.glob("2_*.parquet"):
        match = re.search(r'2_(\d+)\.parquet', file.name)
        if match:
            batch_num = int(match.group(1))
            batch_files.append((batch_num, file))
    
    # Sort by batch number
    batch_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(batch_files)} batch files")
    print(f"Batch range: {batch_files[0][0]} to {batch_files[-1][0]}")
    
    if len(batch_files) == 0:
        print("No batch files found!")
        return False
    
    # Remove output file if it exists
    if output_file.exists():
        print(f"Removing existing output file: {output_file}")
        output_file.unlink()
    
    total_entries = 0
    parquet_writer = None
    
    try:
        print("\nStreaming batches directly to parquet file...")
        
        for i, (batch_num, file_path) in enumerate(tqdm(batch_files, desc="Processing batches")):
            # Read current batch
            batch_df = pd.read_parquet(file_path)
            batch_entries = len(batch_df)
            total_entries += batch_entries
            
            # Convert to pyarrow table
            table = pa.Table.from_pandas(batch_df)
            
            if parquet_writer is None:
                # Create parquet writer with first batch
                parquet_writer = pq.ParquetWriter(output_file, table.schema)
                print(f"Created parquet file with first batch: {batch_entries:,} entries")
            
            # Write current batch to parquet file
            parquet_writer.write_table(table)
            
            # Clear memory immediately
            del batch_df, table
            
            # Progress update every 50 files
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} files, {total_entries:,} entries written")
        
        # Close the parquet writer
        if parquet_writer:
            parquet_writer.close()
            print(f"\nParquet writer closed successfully")
        
        print(f"\n✅ STREAMING MERGE COMPLETED!")
        print(f"Output file: {output_file}")
        print(f"Total entries processed: {total_entries:,}")
        
        # Check file size without loading into memory
        file_size = output_file.stat().st_size / (1024**3)  # GB
        print(f"Output file size: {file_size:.2f} GB")
        print(f"File created successfully at: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during streaming merge: {e}")
        # Clean up writer if it exists
        if parquet_writer:
            try:
                parquet_writer.close()
            except:
                pass
        return False

if __name__ == "__main__":
    success = merge_hplt_2_streaming()
    if not success:
        import sys
        sys.exit(1) 
#!/usr/bin/env python3
"""
HPLT Dataset Processor

This program processes HPLT cleaned dataset files (.jsonl) and converts them to
parquet format, excluding the text field. Features include:
- Batch processing with configurable batch size
- Checkpoint functionality to resume from interruptions
- Progress tracking with multiple progress bars
- Memory efficient processing with resource clearing
- Ordered processing to maintain entry sequence

Usage: python process_hplt_dataset.py <directory>
Example: python process_hplt_dataset.py hplt_cleaned
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import os
import sys
import gc
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional, Tuple
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hplt_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HPLTProcessor:
    """Main processor class for HPLT dataset conversion."""
    
    def __init__(self, input_dir: str, batch_size: int = 100000):
        self.input_dir = Path(input_dir)
        self.batch_size = batch_size
        self.jsonl_files = self._find_jsonl_files()
        self.progress_bars = {}
        
        logger.info(f"Initialized HPLT Processor")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Found {len(self.jsonl_files)} JSONL files")
    
    def _find_jsonl_files(self) -> List[Path]:
        """Find and sort JSONL files in the input directory."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.input_dir}")
        
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        
        # Sort files numerically (1.jsonl, 2.jsonl, etc.)
        def sort_key(path):
            try:
                return int(path.stem)
            except ValueError:
                return float('inf')  # Non-numeric files go to the end
        
        jsonl_files.sort(key=sort_key)
        
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {self.input_dir}")
        
        logger.info(f"Found JSONL files: {[f.name for f in jsonl_files]}")
        return jsonl_files
    
    def _count_lines_in_file(self, file_path: Path) -> int:
        """Count total lines in a JSONL file efficiently with progress bar."""
        logger.info(f"Counting lines in {file_path.name}...")
        
        # Get file size for progress estimation
        file_size = file_path.stat().st_size
        logger.info(f"File size: {file_size / (1024**3):.1f} GB")
        
        # Estimate time based on file size and typical read speeds
        estimated_time_ssd = file_size / (750 * 1024**2)  # 750 MB/s for SSD
        estimated_time_hdd = file_size / (150 * 1024**2)  # 150 MB/s for HDD
        logger.info(f"Estimated counting time: {estimated_time_ssd:.1f}-{estimated_time_hdd:.1f} minutes")
        
        def blocks(file, size=65536):
            """Read file in blocks for efficient processing."""
            while True:
                block = file.read(size)
                if not block:
                    break
                yield block
        
        # Count lines efficiently with progress bar
        with tqdm(
            total=file_size,
            desc=f"Counting lines in {file_path.name}",
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            position=0,
            leave=True
        ) as pbar:
            with open(file_path, 'rb') as f:
                line_count = 0
                bytes_read = 0
                
                for block in blocks(f):
                    line_count += block.count(b'\n')
                    bytes_read += len(block)
                    pbar.update(len(block))
                    
                    # Update progress bar description with current count
                    if bytes_read % (50 * 1024**2) == 0:  # Update every 50MB
                        pbar.set_description(f"Counting lines in {file_path.name} ({line_count:,} lines)")
        
        logger.info(f"File {file_path.name} has {line_count:,} lines")
        return line_count
    
    def _get_output_parquet_path(self, jsonl_file: Path) -> Path:
        """Get the output parquet file path for a given JSONL file."""
        base_name = jsonl_file.stem
        return self.input_dir / f"hplt_cleaned_metadata_{base_name}.parquet"
    
    def _get_batch_directory(self, jsonl_file: Path) -> Path:
        """Get the batch directory path for a given JSONL file."""
        base_name = jsonl_file.stem
        return self.input_dir / f"hplt_{base_name}"
    
    def _get_batch_file_path(self, jsonl_file: Path, batch_num: int) -> Path:
        """Get the batch file path for a given JSONL file and batch number."""
        batch_dir = self._get_batch_directory(jsonl_file)
        return batch_dir / f"{jsonl_file.stem}_{batch_num}.parquet"
    
    def _get_checkpoint_info(self, jsonl_file: Path) -> Tuple[int, int]:
        """
        Get checkpoint information from existing batch files or final parquet file.
        Returns (processed_entries, last_batch_number).
        """
        parquet_path = self._get_output_parquet_path(jsonl_file)
        batch_dir = self._get_batch_directory(jsonl_file)
        
        # Check if final merged file exists
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                processed_entries = len(df)
                last_batch = (processed_entries - 1) // self.batch_size + 1 if processed_entries > 0 else 0
                logger.info(f"Final file found: {processed_entries:,} entries processed")
                return processed_entries, last_batch
            except Exception as e:
                logger.warning(f"Could not read final file {parquet_path}: {e}")
        
        # Check batch files
        if not batch_dir.exists():
            return 0, 0
        
        batch_files = list(batch_dir.glob(f"{jsonl_file.stem}_*.parquet"))
        if not batch_files:
            return 0, 0
        
        # Sort batch files by batch number
        def get_batch_num(path):
            try:
                return int(path.stem.split('_')[-1])
            except ValueError:
                return 0
        
        batch_files.sort(key=get_batch_num)
        
        try:
            processed_entries = 0
            last_batch = 0
            
            for batch_file in batch_files:
                df = pd.read_parquet(batch_file)
                processed_entries += len(df)
                batch_num = get_batch_num(batch_file)
                last_batch = max(last_batch, batch_num)
            
            logger.info(f"Checkpoint found: {processed_entries:,} entries in {len(batch_files)} batch files, last batch {last_batch}")
            return processed_entries, last_batch
        except Exception as e:
            logger.warning(f"Could not read batch files: {e}")
            return 0, 0
    
    def _process_batch(self, batch_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of JSON data and return DataFrame."""
        # Extract all fields except 'text'
        metadata_records = []
        
        for data in batch_data:
            metadata = {key: value for key, value in data.items() if key != 'text'}
            metadata_records.append(metadata)
        
        return pd.DataFrame(metadata_records)
    
    def _save_batch_to_parquet(self, df: pd.DataFrame, batch_file_path: Path, is_first_batch: bool):
        """Save batch DataFrame to individual parquet file."""
        # Create batch directory if it doesn't exist
        batch_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save batch to individual file
        df.to_parquet(batch_file_path, compression='snappy', index=False)
        logger.debug(f"Saved batch to {batch_file_path}")
    
    def _merge_batches_to_final_file(self, jsonl_file: Path):
        """Merge all batch files into a single final parquet file using streaming approach."""
        batch_dir = self._get_batch_directory(jsonl_file)
        final_path = self._get_output_parquet_path(jsonl_file)
        
        # Check if final file already exists
        if final_path.exists():
            logger.info(f"Final file {final_path.name} already exists. Skipping merge.")
            return
        
        # Find all batch files
        batch_files = list(batch_dir.glob(f"{jsonl_file.stem}_*.parquet"))
        
        if not batch_files:
            logger.warning(f"No batch files found in {batch_dir}")
            return
        
        # Sort batch files by batch number to maintain order
        def get_batch_num(path):
            try:
                return int(path.stem.split('_')[-1])
            except ValueError:
                return 0
        
        batch_files.sort(key=get_batch_num)
        
        logger.info(f"Streaming merge of {len(batch_files)} batch files into {final_path.name}")
        
        # Stream batches directly to final parquet file
        total_entries = 0
        parquet_writer = None
        
        try:
            merge_progress = tqdm(
                batch_files, 
                desc=f"Streaming batches for {jsonl_file.stem}",
                position=0,
                unit='file'
            )
            
            for i, batch_file in enumerate(merge_progress):
                try:
                    # Read current batch
                    batch_df = pd.read_parquet(batch_file)
                    batch_entries = len(batch_df)
                    total_entries += batch_entries
                    
                    # Convert to pyarrow table
                    table = pa.Table.from_pandas(batch_df)
                    
                    if parquet_writer is None:
                        # Create parquet writer with first batch
                        parquet_writer = pq.ParquetWriter(final_path, table.schema, compression='snappy')
                        logger.info(f"Created parquet file with first batch: {batch_entries:,} entries")
                    
                    # Write current batch to parquet file
                    parquet_writer.write_table(table)
                    
                    # Update progress
                    merge_progress.set_postfix(entries=batch_entries, total=total_entries)
                    
                    # Clear memory immediately
                    del batch_df, table
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Failed to process batch file {batch_file}: {e}")
                    continue
            
            merge_progress.close()
            
            # Close the parquet writer
            if parquet_writer:
                parquet_writer.close()
                logger.info(f"Parquet writer closed successfully")
            
            if total_entries == 0:
                logger.error("No valid batch files to merge")
                return
            
            logger.info(f"Successfully streamed {len(batch_files)} batches into {final_path}")
            logger.info(f"Final file contains {total_entries:,} entries")
            
            # Check file size
            file_size = final_path.stat().st_size / (1024**3)  # GB
            logger.info(f"Output file size: {file_size:.2f} GB")
            
            # Clean up batch files
            self._cleanup_batch_files(batch_dir)
            
        except Exception as e:
            logger.error(f"Error during streaming merge: {e}")
            # Clean up writer if it exists
            if parquet_writer:
                try:
                    parquet_writer.close()
                except:
                    pass
            # Remove partially created file
            if final_path.exists():
                try:
                    final_path.unlink()
                    logger.info(f"Cleaned up partially created file: {final_path}")
                except:
                    pass
            raise
    
    def _cleanup_batch_files(self, batch_dir: Path):
        """Clean up individual batch files after successful merge."""
        try:
            batch_files = list(batch_dir.glob("*.parquet"))
            logger.info(f"Cleaning up {len(batch_files)} batch files...")
            
            for batch_file in batch_files:
                batch_file.unlink()
            
            # Remove empty batch directory
            if batch_dir.exists() and not list(batch_dir.iterdir()):
                batch_dir.rmdir()
                logger.info(f"Removed empty batch directory: {batch_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup batch files: {e}")
            logger.warning("You may need to manually clean up the batch files")
    
    def _process_single_file(self, jsonl_file: Path, file_index: int, total_files: int):
        """Process a single JSONL file."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file {file_index + 1}/{total_files}: {jsonl_file.name}")
        logger.info(f"{'='*60}")
        
        # Get output path and checkpoint info
        parquet_path = self._get_output_parquet_path(jsonl_file)
        processed_entries, last_batch = self._get_checkpoint_info(jsonl_file)
        
        # Count total lines in file
        total_lines = self._count_lines_in_file(jsonl_file)
        
        if processed_entries >= total_lines:
            logger.info(f"File {jsonl_file.name} already fully processed. Skipping.")
            return
        
        remaining_entries = total_lines - processed_entries
        logger.info(f"Resuming from entry {processed_entries + 1:,} ({remaining_entries:,} remaining)")
        
        # Create progress bars
        file_desc = f"Part {jsonl_file.stem}"
        self.progress_bars['file'] = tqdm(
            total=total_files,
            desc="Overall Progress",
            position=0,
            initial=file_index
        )
        
        self.progress_bars['entries'] = tqdm(
            total=total_lines,
            desc=f"{file_desc} Entries",
            position=1,
            initial=processed_entries,
            unit='entries'
        )
        
        # Process file in batches
        batch_data = []
        current_batch = last_batch + 1
        line_count = 0
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    
                    # Skip already processed entries
                    if line_count <= processed_entries:
                        continue
                    
                    try:
                        data = json.loads(line.strip())
                        batch_data.append(data)
                        
                        # Update progress
                        self.progress_bars['entries'].update(1)
                        
                        # Process batch when full
                        if len(batch_data) >= self.batch_size:
                            self._process_and_save_batch(
                                batch_data, jsonl_file, current_batch, 
                                file_desc, processed_entries == 0 and current_batch == 1
                            )
                            
                            # Clear batch data and force garbage collection
                            batch_data.clear()
                            gc.collect()
                            current_batch += 1
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_count}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_count}: {e}")
                        continue
                
                # Process remaining data in final batch
                if batch_data:
                    self._process_and_save_batch(
                        batch_data, jsonl_file, current_batch, 
                        file_desc, processed_entries == 0 and current_batch == 1
                    )
        
        finally:
            # Clean up progress bars
            if 'batch' in self.progress_bars:
                self.progress_bars['batch'].close()
            self.progress_bars['entries'].close()
            
            # Update file progress
            self.progress_bars['file'].update(1)
        
        # Merge all batches into final file
        logger.info(f"Merging batches for {jsonl_file.name}...")
        self._merge_batches_to_final_file(jsonl_file)
        
        logger.info(f"Completed processing {jsonl_file.name}")
        logger.info(f"Output: {parquet_path}")
    
    def _process_and_save_batch(self, batch_data: List[Dict], jsonl_file: Path, 
                               batch_num: int, file_desc: str, is_first_batch: bool):
        """Process and save a batch of data."""
        # Create/update batch progress bar
        if 'batch' not in self.progress_bars:
            self.progress_bars['batch'] = tqdm(
                total=self.batch_size,
                desc=f"{file_desc} Batch {batch_num}",
                position=2,
                unit='entries'
            )
        else:
            self.progress_bars['batch'].reset()
            self.progress_bars['batch'].set_description(f"{file_desc} Batch {batch_num}")
        
        # Process batch with progress updates
        processed_count = 0
        chunk_size = max(1, len(batch_data) // 10)  # Update progress 10 times per batch
        
        for i in range(0, len(batch_data), chunk_size):
            chunk = batch_data[i:i + chunk_size]
            processed_count += len(chunk)
            self.progress_bars['batch'].update(len(chunk))
            
            # Small delay to make progress visible
            time.sleep(0.01)
        
        # Process the batch
        df = self._process_batch(batch_data)
        
        # Save to parquet
        self._save_batch_to_parquet(df, self._get_batch_file_path(jsonl_file, batch_num), is_first_batch)
        
        # Log batch completion
        logger.info(f"Batch {batch_num} completed: {len(batch_data)} entries processed")
        
        # Clear DataFrame and force garbage collection
        del df
        gc.collect()
    
    def process_all_files(self):
        """Process all JSONL files in the input directory."""
        logger.info(f"Starting processing of {len(self.jsonl_files)} files")
        
        start_time = time.time()
        
        try:
            for file_index, jsonl_file in enumerate(self.jsonl_files):
                self._process_single_file(jsonl_file, file_index, len(self.jsonl_files))
            
            # Close overall progress bar
            if 'file' in self.progress_bars:
                self.progress_bars['file'].close()
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"ALL FILES PROCESSED SUCCESSFULLY!")
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            logger.info(f"{'='*60}")
            
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user. Progress saved.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process HPLT dataset files and convert to parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_hplt_dataset.py hplt_cleaned
  python process_hplt_dataset.py /path/to/hplt_cleaned
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory containing JSONL files to process'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100000,
        help='Number of entries to process in each batch (default: 100000)'
    )
    
    args = parser.parse_args()
    
    try:
        processor = HPLTProcessor(args.directory, args.batch_size)
        processor.process_all_files()
        
    except FileNotFoundError as e:
        logger.error(f"Directory or file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
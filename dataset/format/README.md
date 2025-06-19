# `format`

Each of these programs will read through the entire directory created in the previous step and will look for all `.jsonl` files. For each file found, the script will:

1. **Resume or start processing** from the last checkpoint, if available.  
2. **Stream and parse** each JSON line to extract the target field (e.g., `warc_headers.warc-target-uri`).  
3. **Aggregate counts** in memory, periodically checkpointing progress.  
4. **Emit intermediate results** as per-file `.parquet` in the `merged/` folder.  
5. **After all files are processed**, merge all intermediate Parquet files into a final, sorted output.  

### create_oscar_parquet.py
```bash
python3 create_oscar_parquet.py oscar
```

### create_hplt_parquet.py
```bash
python3 create_hplt_parquet.py hplt
```

## Output Structure

```text
<dataset>_domains/
├── checkpoints/           # JSON files recording processed line counts
├── merged/                # Intermediate per-file .parquet results
└── progress.json          # Tracks which files have completed

<dataset>_domains.parquet  # Final merged domain frequency Parquet file
```

###
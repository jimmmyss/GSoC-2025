# `format`

Each of these programs will read through the entire directory created in the previous step and will look for all `.jsonl` files. For each file found, the script will:

1. **Resume or start processing** from the last checkpoint, if available.  
2. **Stream and parse** each JSON line to extract the target field.  
3. **Aggregate counts** in memory, periodically checkpointing progress after 100000 lines.  
4. **Write Per‑File Outputs** as per-file `.parquet` in the `merged/` folder.
5. **After all files are processed**, load all the per‑file Parquet files, sum domain counts across them, sort by frequency, write the definitive `oscar_domains.parquet` file and remove the temporary `<dataset>_domains` directory.

## How to run each formatter

### create_oscar_parquet.py
```bash
python3 create_oscar_parquet.py oscar
```

### create_hplt_parquet.py
```bash
python3 create_hplt_parquet.py hplt
```

## Output

### Output format

| domain             | count   |
|--------------------|---------|
| google.com         | 100000  |
| youtube.com        | 80000   |
| facebook.com       | 60000   |
| wikipedia.org      | 40000   |
| reddit.com         | 30000   |
| amazon.com         | 25000   |
| twitter.com        | 20000   |
| stackoverflow.com  | 15000   |
| bing.com           | 12000   |
| ...                | ...     |

### Output structure

```text
<dataset>_domains/
├── checkpoints/           # JSON files recording processed line counts
├── merged/                # Intermediate per-file .parquet results
└── progress.json          # Tracks which files have completed

<dataset>_domains.parquet  # Final merged domain frequency Parquet file
```
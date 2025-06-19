# `dataset` folder overview

The `dataset` directory contains the four sequential stages of the data‑processing pipeline:

### 1. `download`  
- Fetch datasets from [OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/tree/main/el_meta) and [HPLT](https://hplt-project.org/datasets/v2.0) in compressed `.zst` format.  
- Decompress the `.zst` files.

### 2. `format`  
1. Resume or start processing from the last checkpoint, if available.  
2. Stream and parse each JSON line to extract the `warc_headers.warc-target-uri` field.  
3. Aggregate domain counts in memory, checkpointing progress every 100,000 lines.  
4. Write per-file outputs as `.parquet` files in the `merged/` directory.  
5. Merge all outputs by summing domain counts across `.parquet` files, sorting by frequency, writing the final `oscar_domains.parquet`, and cleaning up temporary directories.

### 3. `scrape`  
- Enrich records by scraping metadata such as:  
  - Web page titles  
  - HTTP headers  
  - Linked assets  
  - Content types and page length

### 4. `categorize`  
- Assign categories using:  
  - Rule-based heuristics  
  - Classification models  
  - Language or topic filters for downstream curation
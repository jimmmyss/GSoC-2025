# `dataset` folder overview

The `dataset` directory contains the four sequential stages of the data‑processing pipeline:

### 1. `download`  
- Fetch datasets from [OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/tree/main/el_meta) and [HPLT](https://hplt-project.org/datasets/v2.0) in compressed `.zst` format.  
- Decompress the `.zst` files.

### 2. `format`  
- Parse each `.jsonl` file to extract all domain names and how many times each domain appears across the dataset.  
- Write the final `<dataset>_domains.parquet` that contains a deduplicated, sorted list of domains.

### 3. `scrape`  
- Enrich records by scraping the following metadata:  
  - `status_code`
  - `title`
  - `meta_description`
  - `keywords` 
  - Open Graph Metadata
- Write the final `<dataset>_domains_metadata.parquet` that contains all of the metadata scraped for each domain.

### 4. `classify`  
- Assign categories using:  
  - Rule-based heuristics  
  - Classification models  
  - Language or topic filters for downstream curation
   Applies classification models or rule‑based logic to assign topic or quality categories to each JSON object for downstream filtering and analysis.
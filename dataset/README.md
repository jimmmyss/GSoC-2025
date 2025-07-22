# `dataset`

The `dataset` directory contains the four sequential stages of the data‑processing pipeline:

### 1. `download`  
- Fetch datasets from [OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/tree/main/el_meta) and [HPLT](https://hplt-project.org/datasets/v2.0) in compressed `.zst` format.  
- Decompress the `.zst` files.

### 2. `format`
- Process each `.jsonl` file to extract all the necessary data.
- Two alternative formatting paths are supported:

  **a. Metadata format**
  - Extract domain names and how many times each domain appears across the dataset, sorted by frequency.
  - Write output to `<dataset>_domains.parquet`.

  **b. Landing page format**
  - Extract all dataset entries based on the base links.
  - Write output to `<dataset>_domains_landing_pages.parquet`.

### 3. `scrape` (only for metadata)
- Enrich records by scraping the following metadata:  
  - `status_code`
  - `title`
  - `meta_description`
  - `keywords` 
  - Open Graph Metadata
- Write the output to `<dataset>_domains_metadata.parquet` that contains all of the scraped metadata for each domain.

### 4. `classify` (only for metadata)
- Assign categories using a 6-model classification pipeline and gemini.
- Write output to `<dataset>_domains_metadata_categories.parquet`.

### 5. `analysis` (optional)
- Automated topic modeling for domain category discovery with BERTopic.
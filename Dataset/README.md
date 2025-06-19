# `dataset` folder overview

The `dataset` directory contains the four sequential stages of the data‑processing pipeline:

## 1.`download`  
   Fetches datasets from [Oscar](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/tree/main/el_meta) & [HPLT](https://hplt-project.org/datasets/v2.0) in compressed `.zst` format and decompresses them.

## 2.`format`
   Reads the `.jsonl` archives, extracts all domain names from every link, aggregates how many times each domain appears, and converts each record into the standardized `.parquet` schema.

## 3. `scrape`
   Enriches each JSON record by scraping additional metadata (e.g., web page titles, HTTP headers, linked assets).

## 4. `categorize`
   Applies classification models or rule‑based logic to assign topic or quality categories to each JSON object for downstream filtering and analysis.
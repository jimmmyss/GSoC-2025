## `dataset` Folder Overview

The `dataset` directory contains the four sequential stages of our data‑processing pipeline:

1. **downloader**  
   Fetches raw data files from external sources (e.g., Hugging Face, APIs) and saves them in compressed `.zst` format.

2. **format**  
   Reads the downloaded archives, decompresses them, and converts each record into the standardized JSON schema (Oscar or HPLT).

3. **scraper**  
   Enriches each JSON record by scraping additional metadata (e.g., web page titles, HTTP headers, linked assets).

4. **categorizer**  
   Applies classification models or rule‑based logic to assign topic or quality categories to each JSON object for downstream filtering and analysis.

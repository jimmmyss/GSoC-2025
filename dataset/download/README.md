# Dataset Downloaders

This folder contains dataset downloaders for [Oscar](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/tree/main/el_meta) & [HPLT](https://hplt-project.org/datasets/v2.0).

## How to run each downloader

### download_oscar.py
```bash
python download_oscar.py <hugging_face_api_token>
```

### download_hplt.py
```bash
python download_hplt.py
```

## How to decompress the datasets

After running each downloader script, a compressed `.zst` file will be created in the new corresponding output directories. These file use the `Zstandard` compression format and needs to be decompressed before you can use the dataset. To decompress these files you need to run:

### Decompress
```bash
cd <directory>
find . -name "*.zst" -type f -exec zstd -d --rm {} \;
cd ..
```

## Dataset format

Below are the JSON schemas for the `Oscar` and `HPLT` datasets. Each schema represents a standalone JSON object conforming to its respective format:

### Οscar
```bash
{
  "content": "string",                    // Main text content (very long)
  
  "warc_headers": {                      // Web Archive (WARC) metadata
    "content-length": "string",          // Size in bytes ("1086")
    "warc-refers-to": "string",          // UUID reference 
    "warc-identified-content-language": "string",  // Languages ("eng,ell")
    "warc-record-id": "string",          // Unique record UUID
    "content-type": "string",            // MIME type ("text/plain")
    "warc-target-uri": "string",         // Original URL that was crawled
    "warc-block-digest": "string",       // SHA1 hash for content verification
    "warc-date": "string",               // When crawled ("2022-11-27T12:56:25Z")
    "warc-type": "string"                // WARC record type ("conversion")
  },
  
  "metadata": {                          // Processing metadata
    "identification": {                  // Primary language detection
      "label": "string",                 // Language code ("el" = Greek)
      "prob": number                     // Confidence score (0.7279648)
    },
    
    "harmful_pp": number,                // Harmful content perplexity score
    
    "tlsh": "string",                    // Fuzzy hash for near-duplicate detection
    
    "quality_warnings": null,            // Quality issues (null = none found)
    
    "categories": null,                  // Content categories (null = none assigned)
    
    "sentence_identifications": [        // Per-sentence language detection
      {
        "label": "string",               // Language per sentence ("en", "el")
        "prob": number                   // Confidence per sentence
      }
    ]
  }
}
```

### hplt
```bash
{
  "f": "string",                    // File source path
  "o": number,                      // Offset (3408300706)
  "s": number,                      // Size (104343)
  "rs": number,                     // Related size (374173)
  "u": "string",                    // URL
  "c": "string",                    // Content type ("text/html")
  "ts": "string",                   // Timestamp ("2023-10-13T03:53:51Z")
  "collection": "string",           // Collection name ("archivebot")
  "lang": ["string", "string"],     // Language codes array
  "prob": [number, number],         // Probability scores array
  "text": "string",                 // Main text content (very long)
  "seg_langs": ["string"],          // Segment languages array
  "robotstxt": "string",            // Robots.txt status ("allowed")
  "id": "string",                   // Unique identifier (hash)
  "filter": "string",               // Filter status ("keep")
  "pii": [],                        // Personal info array (empty)
  "doc_scores": [number, number]    // Document quality scores array
}
```



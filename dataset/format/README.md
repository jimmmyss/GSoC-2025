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
| wikipedia.com      | 40000   |
| google.com         | 10000   |
| youtube.com        | 5000    |
| facebook.com       | 2000    |
| ...                | ...     |

### Output structure

```text
<dataset>_domains/
├── checkpoints/           # JSON files recording processed line counts
├── merged/                # Intermediate per-file .parquet results
└── progress.json          # Tracks which files have completed

<dataset>_domains.parquet  # Final merged domain frequency Parquet file
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


# `format`

## Metadata format
Each of these programs will read through the entire directory created in the previous step and will look for all `.jsonl` files. For each file found, the script will:

1. **Resume or start processing** from the last checkpoint, if available.  
2. **Stream and parse** each JSON line to extract the target field.  
3. **Aggregate counts** in memory, periodically checkpointing progress after 100000 lines.  
4. **Write Per‑File Outputs** as per-file `.parquet` in the `merged/` folder.
5. **After all files are processed**, load all the per‑file Parquet files, sum domain counts across them, sort by frequency, write the definitive `oscar_domains.parquet` file and remove the temporary `<dataset>_domains` directory.

## Landing page format

This program will read through the entire directory created in the previous and processes JSONL and will look for all `.jsonl` files. For each file found, the script will:

1. **Process files in numerical order** (1.jsonl, 2.jsonl, etc.) with resume capability.
2. **Stream and parse** each JSON line to extract URL and text content.
3. **Filter for landing pages** - keeps only domain-level URLs and landing page paths.
4. **Batch processing** of 100,000 records with memory management and garbage collection.
5. **Output single parquet file** with all original fields plus text character count field.

**Landing page types filtered:**
- **Domain-level URLs**: https://example.com, https://example.com/
- **Landing page paths**: /home, /index.html, /main.php, /welcome, /dashboard, etc.

## How to run each formatter

### format_metadata_oscar.py
```bash
python3 format_metadata_oscar.py <oscar_directory>
```

### format_metadata_hplt.py
```bash
python3 format_metadata_hplt.py <hplt_directory>
```

### format_landing_page_hplt.py
```bash
python3 format_landing_page_hplt.py <hplt_directory>
```

## Output

### Metadata output format

| domain             | count   |
|--------------------|---------|
| wikipedia.com      | 40000   |
| google.com         | 10000   |
| youtube.com        | 5000    |
| facebook.com       | 2000    |
| ...                | ...     |

### Landing page output format

| f | o | s | rs | u | c | ts | collection | lang | prob | text | seg_langs | robotstxt | id | filter | pii | doc_scores | text_char_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| /data/file1.warc | 3408300706 | 104343 | 374173 | https://wikipedia.org/ | text/html | 2023-10-13T03:53:51Z | archivebot | ["en"] | [0.95] | "Wikipedia is a free online encyclopedia..." | ["en"] | allowed | abc123def456 | keep | [] | [0.87, 0.92] | 2456 |
| /data/file2.warc | 2156890123 | 87234 | 298456 | https://github.com/ | text/html | 2023-10-13T04:12:33Z | archivebot | ["en"] | [0.98] | "GitHub is where over 100 million developers..." | ["en"] | allowed | def789ghi012 | keep | [] | [0.91, 0.89] | 1823 |
| /data/file3.warc | 5672341890 | 156789 | 445621 | https://example.com/home | text/html | 2023-10-13T05:23:17Z | archivebot | ["en"] | [0.88] | "Welcome to our homepage. We provide..." | ["en"] | allowed | ghi345jkl678 | keep | [] | [0.76, 0.84] | 3201 |
| /data/file4.warc | 1234567890 | 92156 | 312098 | https://stackoverflow.com/ | text/html | 2023-10-13T06:45:29Z | archivebot | ["en"] | [0.96] | "Stack Overflow is the largest community..." | ["en"] | allowed | jkl901mno234 | keep | [] | [0.93, 0.88] | 1967 |
| /data/file5.warc | 7890123456 | 134567 | 398745 | https://reddit.com/index | text/html | 2023-10-13T07:18:52Z | archivebot | ["en"] | [0.89] | "Reddit is a network of communities..." | ["en"] | allowed | mno567pqr890 | keep | [] | [0.81, 0.86] | 2734 |

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
# `scrape`

After the creation of each individual `.parquet` file, the scraper will go through **each domain** and extract the following metadata from the corresponding web pages:

### 1. `status_code`
- The HTTP status code returned by the server.
- Used to determine if the request was successful (`200`), redirected (`301`, `302`), or failed (`404`, `500`, etc.).

### 2. `title`
- The content of the `<title>` tag in the HTML document.
- Represents the name of the page shown in the browser tab.

### 3. `meta_description`
- Extracted from the `<meta name="description" content="...">` tag.
- A summary of the page’s content shown in the browser tab.

### 4. `keywords`
- Extracted from the `<meta name="keywords" content="...">` tag.
- Contains comma-separated keywords relevant to the page (largely deprecated in modern SEO).

### 5. Open Graph Metadata

These values are extracted based on the [Open Graph protocol](https://ogp.me/), which is used to optimize content sharing on social media platforms.

- **`og_title`**
  - Extracted from the `<meta property="og:title" content="...">` tag.
  - Defines how the page title appears when shared.

- **`og_description`**
  - Extracted from the `<meta property="og:description" content="...">` tag.
  - Provides a social-media-optimized summary of the page content.

- **`og_type`**
  - Extracted from the `<meta property="og:type" content="...">` tag.
  - Specifies the type of object (e.g., `website`, `article`, `video.movie`, etc.).

## How to run scraper

### domain_scraper.py
```bash
python3 domain_scraper.py oscar.parquet
```

### Output format

| domain            | count  | status_code | title | meta_description | keywords | og_title | og_description | og_type |
|-------------------|--------|-------------|-------|------------------|----------|----------|----------------|---------|
| wikipedia.com     | 40000  |             |       |                  |          |          |                |         |
| google.com        | 10000  |             |       |                  |          |          |                |         |
| youtube.com       | 5000   |             |       |                  |          |          |                |         |
| facebook.com      | 2000   |             |       |                  |          |          |                |         |
| ...               | ...    |             |       |                  |          |          |                |         |

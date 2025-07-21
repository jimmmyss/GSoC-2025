# `classify`

## 1. 6-Model classification (`classifiy.py`)
Uses an **ensemble voting system** with six different embedding models. Each model independently analyzes the domain metadata and votes for the most appropriate category. The final classification is determined by **majority vote** among the six models, with advanced conflict resolution for tie-breaking situations.

**How it works:**
1. **Six models vote**: Each embedding model processes the domain metadata and predicts a category
2. **Majority wins**: The category with the most votes becomes the final prediction  
3. **Conflict resolution**: For ties, the system retries with text variations and applies domain-specific rules
4. **High accuracy**: Multiple models reduce individual model bias and increase overall accuracy

**Models Used:**
- [`multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
- [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)  
- [`LaBSE`](https://huggingface.co/sentence-transformers/LaBSE)  
- [`jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)
- [`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)

**Key Features:**
- **Ensemble Voting** – Six state-of-the-art multilingual embedding models vote independently
- **Conflict Resolution** – Retries up to 3 times if prediction disagreement is too high  
- **Confidence Scoring** – Outputs numerical confidence score per domain prediction  
- **Rule Enhancement** – Post-processing rules correct and reinforce model outputs  
- **Vague Detection** – "Other" assigned only when content is genuinely unclear

## 2. Metadata-Based Gemini AI (`classify_gemini_metadata.py`)
**Simply asks Gemini LLM which category best fits the available metadata.** This approach extracts metadata fields from existing parquet files (domain name, title, meta description, keywords, etc.) and presents them to Google's Gemini 2.5 Flash model for direct categorization.

**How it works:**
1. **Extract metadata**: Reads domain, title, meta_description, keywords, og_title, og_description from parquet files
2. **Combine fields**: Merges all available metadata into a single text prompt  
3. **Ask Gemini**: Sends the metadata to Gemini with a categorization prompt
4. **Get response**: Gemini directly responds with the most appropriate category

**Key Features:**
- **Smart Field Detection** – Automatically detects metadata columns with name variations and positional fallbacks
- **Multi-Field Support** – Uses domain, title, meta_description, keywords, og_title, og_description, og_type, redirect_url
- **Gemini 2.5 Flash API** – Uses Google's latest language model with rate limiting and retry logic
- **Batch Processing** – Processes domains in configurable batches (default: 100) with memory management
- **Resume Capability** – Automatically resumes from last completed batch after interruptions

## 3. Live Web Scraping with Gemini AI (`classify_gemini_landing_page.py`)
**Visits actual landing page content and decides the category based on live webpage analysis.** This approach fetches real webpage content by making HTTP requests to domains, extracts meaningful content from the HTML, and asks Gemini to categorize based on the actual live website content.

**How it works:**
1. **Visit websites**: Makes HTTP/HTTPS requests to access actual webpage content
2. **Extract content**: Parses HTML to get title, meta tags, main content, and Open Graph data
3. **Clean content**: Processes and truncates content for optimal LLM analysis
4. **Ask Gemini**: Sends the live content to Gemini for categorization based on actual webpage

**Key Features:**
- **Live Website Access** – Directly fetches content via HTTP/HTTPS with fallback URL strategies (www, http/https)
- **Content Extraction** – Parses HTML for title, meta description, keywords, Open Graph data, and main page content
- **Gemini 2.5 Flash Integration** – Uses Google's latest AI model with built-in rate limiting and error handling
- **Individual Processing** – Each domain classified individually with immediate saving and resume capability
- **Progress Tracking** – Real-time progress bars showing current domain and assigned category

## Predefined Categories

- **Ηλεκτρονικό Εμπόριο & Αγορές** (E-Commerce & Shopping)  
- **Ειδήσεις & Μέσα Ενημέρωσης** (News & Media)  
- **Κοινωνικά Δίκτυα & Κοινότητα** (Social Media & Community)  
- **Τεχνολογία & Λογισμικό** (Technology & Software)  
- **Ψυχαγωγία & Μέσα** (Entertainment & Media)  
- **Εκπαίδευση & Έρευνα** (Education & Research)  
- **Υγεία & Ιατρική** (Health & Medical)  
- **Κυβέρνηση & Δημόσιες Υπηρεσίες** (Government & Public Services)  
- **Ταξίδια & Τουρισμός** (Travel & Tourism)  
- **Χρηματοοικονομικά & Τραπεζικά** (Finance & Banking)  
- **Αθλητισμός & Αναψυχή** (Sports & Recreation)  
- **Άλλο** (Other)

## How to run each classifier

### classifiy.py
```bash
python classifiy.py <dataset>.parquet
```

### classify_gemini_metadata.py
```bash
python classify_gemini_metadata.py <dataset>.parquet
```

### classify_gemini_landing_page.py
```bash
python classify_gemini_landing_page.py <dataset>.parquet
```

## Output Format

| domain | title | meta_description | keywords | category | confidence |
|---------|-------|------------------|----------|----------|------------|
| wikipedia.org | Wikipedia | Free encyclopedia | knowledge, reference | Education & Research | 0.92 |
| amazon.com | Amazon | Online shopping | shopping, ecommerce | E-Commerce & Shopping | 0.89 |
| github.com | GitHub | Developer platform | code, programming | Technology & Software | 0.94 |
| cnn.com | CNN | Breaking news | news, current events | News & Media | 0.88 |
| netflix.com | Netflix | Streaming service | movies, entertainment | Entertainment & Media | 0.91 |
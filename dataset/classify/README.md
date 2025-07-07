# `classify`

A sophisticated domain classification system that automatically categorizes websites/domains into meaningful categories using state-of-the-art machine learning models.

The Enhanced Domain Classifier takes metadata from websites—like domain name, title, meta description, and keywords—and classifies them into **12 predefined categories** using a **multi-model ensemble of language models** combined with rule-based enhancements.

## Predefined Categories

- **E-Commerce & Shopping**
- **News & Media**
- **Social Media & Community**
- **Technology & Software**
- **Entertainment & Media**
- **Education & Research**
- **Health & Medical**
- **Government & Public Services**
- **Travel & Tourism**
- **Finance & Banking**
- **Sports & Recreation**
- **Other**

## Multi-Model Ensemble Classification

Uses six state-of-the-art multilingual embedding models to ensure accurate predictions across languages:

- [`multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
- [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)  
- [`LaBSE`](https://huggingface.co/sentence-transformers/LaBSE)  
- [`jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)
- [`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)

Each model generates vector embeddings used for downstream classification.

### Advanced Classification Logic

- **Conflict Resolution** – Retries up to 3 times if prediction disagreement is too high  
- **Confidence Scoring** – Outputs a numerical confidence score per domain prediction  
- **Rule-Based Enhancement** – Post-processing rules correct and reinforce model outputs  
- **Vague Content Detection** – “Other” is assigned only when content is genuinely unclear  

### Robust Processing

- **Batch Processing** – Classifies domains in batches for scalability  
- **Resume Capability** – Automatically resumes from last processed point  
- **Memory Management** – Actively clears memory and reclaims resources  
- **Progress Tracking** – Real-time progress bars and logs  

### Multilingual Support

- **Greek Category Support** – Internal logic uses Greek labels for Greek sites  
- **English Output** – All outputs translated to English for broad compatibility  
- **Multilingual Models** – Full language coverage across supported models  

## How to run scraper

### classify.py
```bash
python classify.py <dataset>.parquet
```

## Output format

| domain            | count  | status_code | title | meta_description | keywords | category | confidence |
|-------------------|--------|-------------|-------|------------------|----------|----------|------------|
| wikipedia.com     | 40000  |             |       |                  |          |          |            |
| google.com        | 10000  |             |       |                  |          |          |            |
| youtube.com       | 5000   |             |       |                  |          |          |            |
| facebook.com      | 2000   |             |       |                  |          |          |            |
| ...               | ...    |             |       |                  |          |          |            |

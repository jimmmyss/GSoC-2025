# `classify`

A sophisticated domain classification system that automatically categorizes websites/domains into meaningful categories using state-of-the-art machine learning models.

The Enhanced Domain Classifier takes metadata from websites—like domain name, title, meta description, and keywords—and classifies them into **12 predefined categories** using a **multi-model ensemble of language models** combined with rule-based enhancements.

## Predefined Categories

- **E-Commerce & Shopping** – Online stores, marketplaces, retail sites  
- **News & Media** – News outlets, journalism, press sites  
- **Social Media & Community** – Social networks, forums, chat platforms  
- **Technology & Software** – Tech companies, programming, dev tools  
- **Entertainment & Media** – Movies, music, games, streaming services  
- **Education & Research** – Universities, academic content, learning platforms  
- **Health & Medical** – Healthcare, hospitals, medical info  
- **Government & Public Services** – Government agencies, official portals  
- **Travel & Tourism** – Hotels, flights, booking platforms  
- **Finance & Banking** – Banking, investments, financial services  
- **Sports & Recreation** – Sports, fitness, athletic sites  
- **Other** – Only used when the domain content is truly ambiguous

## Multi-Model Ensemble Classification

Uses six state-of-the-art multilingual embedding models to ensure accurate predictions across languages:

- `E5-Large` (multilingual)  
- `MPNet` (paraphrase-multilingual)  
- `MiniLM` (lightweight multilingual)  
- `LaBSE` (language-agnostic BERT)  
- `Jina Embeddings v3`
- `mDeBERTa` (multilingual)  

Each model generates vector embeddings used for downstream classification.

### dvanced Classification Logic

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
python enhanced_domain_classifier.py <dataset>.parquet
```

## Output format

| domain            | count  | status_code | title | meta_description | keywords | category | confidence |
|-------------------|--------|-------------|-------|------------------|----------|----------|------------|
| wikipedia.com     | 40000  |             |       |                  |          |          |            |
| google.com        | 10000  |             |       |                  |          |          |            |
| youtube.com       | 5000   |             |       |                  |          |          |            |
| facebook.com      | 2000   |             |       |                  |          |          |            |
| ...               | ...    |             |       |                  |          |          |            |

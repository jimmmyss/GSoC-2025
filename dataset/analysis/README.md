# `analysis`

## Metadata analysis

**Automatically discovers hidden domain categories from metadata without any predefined categories.** This approach uses unsupervised machine learning to analyze domain metadata, cluster similar domains together, and generate meaningful category names based on discovered patterns.

**How it works:**
1. **Load metadata**: Extracts domain name, title, and meta description from parquet files
2. **Create embeddings**: Uses multilingual SentenceTransformer to convert text to vector embeddings
3. **Cluster domains**: Applies UMAP dimensionality reduction and HDBSCAN clustering to find natural groups
4. **Generate categories**: Uses BERTopic to create topic representations and automatically name discovered categories
5. **Save results**: Outputs discovered categories in multiple formats with keywords and statistics

**Key Features:**
- **Unsupervised Discovery** – No predefined categories required, discovers patterns naturally from data
- **Advanced ML Pipeline** – Combines SentenceTransformer, UMAP, HDBSCAN, and BERTopic for robust clustering
- **Automatic Naming** – Generates meaningful category names from discovered keywords and patterns  
- **Multi-Format Output** – Saves results as CSV, JSON, and human-readable text summary
- **GPU Acceleration** – Supports CUDA for faster embedding computation on large datasets
- **Progress Tracking** – Detailed progress bars and statistics throughout the discovery process

**Models Used:**
- **SentenceTransformer**: `intfloat/multilingual-e5-base` for text embeddings
- **UMAP**: Dimensionality reduction (15 neighbors, 5 components, cosine metric)
- **HDBSCAN**: Clustering (min 20 docs per cluster, euclidean metric)
- **CountVectorizer**: Text vectorization (1-2 grams, max 5000 features)
- **BERTopic**: Topic modeling with KeyBERT and MaximalMarginalRelevance representation

## Output Files

The system creates a `domain_categories/` directory with three output formats:

### 1. `discovered_categories.csv`
| topic_id | name | document_count | keywords | all_keywords |
|----------|------|----------------|----------|--------------|
| 0 | News & Media Related | 15,230 | ['news', 'breaking', 'media', 'journalism', 'reporter'] | ['news', 'breaking', 'media', 'journalism', 'reporter', 'article', 'press', 'story'] |
| 1 | Technology & Software | 12,845 | ['technology', 'software', 'programming', 'developer', 'code'] | ['technology', 'software', 'programming', 'developer', 'code', 'github', 'api', 'platform'] |
| 2 | Education & Research | 9,567 | ['education', 'university', 'research', 'academic', 'learning'] | ['education', 'university', 'research', 'academic', 'learning', 'school', 'knowledge', 'study'] |

### 2. `categories_summary.txt`
```
DISCOVERED DOMAIN CATEGORIES
==================================================

Analysis Date: 2024-01-15 14:30:25
Total Categories: 12

 1. News & Media Related
    Documents: 15,230
    Keywords: news, breaking, media, journalism, reporter

 2. Technology & Software
    Documents: 12,845
    Keywords: technology, software, programming, developer, code
```

### 3. `categories.json`
```json
{
  "analysis_date": "2024-01-15T14:30:25",
  "total_categories": 12,
  "categories": [
    {
      "topic_id": 0,
      "name": "News & Media Related",
      "document_count": 15230,
      "keywords": ["news", "breaking", "media", "journalism", "reporter"],
      "all_keywords": ["news", "breaking", "media", "journalism", "reporter", "article", "press", "story"]
    }
  ]
}
```

## Discovery Process

### Phase 1: Model Setup
- Loads multilingual embedding model with GPU/CPU detection
- Configures UMAP for dimensionality reduction
- Sets up HDBSCAN clustering parameters
- Initializes text vectorizer and BERTopic pipeline

### Phase 2: Data Processing
- Loads parquet file and extracts valid text features
- Combines domain name, title, and meta description
- Filters out entries with insufficient text content
- Reports extraction statistics and data quality

### Phase 3: Embedding Generation
- Computes vector embeddings in batches (500 texts per batch)
- Uses multilingual SentenceTransformer for semantic understanding
- Progress tracking for large datasets

### Phase 4: Topic Discovery  
- Applies UMAP to reduce embedding dimensions
- Uses HDBSCAN to identify natural clusters
- Generates topic representations with BERTopic
- Reports number of discovered topics and outliers

### Phase 5: Category Naming
- Extracts keywords from each discovered topic
- Generates meaningful category names automatically
- Ranks categories by document count
- Creates comprehensive topic metadata

## Landing page analysis

## How to run

```bash
python3 analysis_metadata.py <parquet_file>
```

```bash
python3 analysis_landing_page.py <parquet_file>
```
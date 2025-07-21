#!/usr/bin/env python3
"""
BERTopic Domain Category Discovery
Automatically discovers domain categories from parquet metadata files
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# BERTopic and ML imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

class DomainCategoryDiscovery:
    """Main class for discovering domain categories using BERTopic"""
    
    def __init__(self):
        self.embedding_model = None
        self.topic_model = None
        
    def setup_models(self):
        """Initialize all models with progress tracking"""
        print("🔧 Setting up models...")
        
        with tqdm(total=5, desc="Initializing models") as pbar:
            # 1. Setup embedding model
            pbar.set_description("Loading embedding model")
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)
            print(f"   📱 Using device: {device}")
            pbar.update(1)
            
            # 2. Setup UMAP
            pbar.set_description("Configuring UMAP")
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            pbar.update(1)
            
            # 3. Setup HDBSCAN
            pbar.set_description("Configuring HDBSCAN")
            hdbscan_model = HDBSCAN(
                min_cluster_size=20,
                min_samples=5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            pbar.update(1)
            
            # 4. Setup vectorizer
            pbar.set_description("Configuring vectorizer")
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=5000,
                min_df=3,
                max_df=0.95
            )
            pbar.update(1)
            
            # 5. Create BERTopic model
            pbar.set_description("Creating BERTopic model")
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=[KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)],
                calculate_probabilities=False,  # Disable to avoid memory issues
                verbose=False
            )
            pbar.update(1)
        
        print("✅ All models configured successfully")
    
    def clean_domain(self, domain_text):
        """Clean and normalize domain text"""
        if not domain_text or domain_text in ['nan', 'None', '', 'null']:
            return None
            
        domain = str(domain_text).strip().lower()
        
        # Remove protocols
        domain = domain.replace('https://', '').replace('http://', '')
        domain = domain.replace('www.', '').replace('m.', '')
        
        # Remove paths, ports, queries
        domain = domain.split('/')[0].split(':')[0].split('?')[0].split('#')[0]
        
        # Remove common subdomains but keep meaningful ones
        parts = domain.split('.')
        if len(parts) > 2:
            subdomain = parts[0]
            meaningful = ['news', 'blog', 'shop', 'store', 'mail', 'en', 'el', 'gr', 'app', 'api']
            if subdomain not in meaningful:
                domain = '.'.join(parts[1:])
        
        return domain if len(domain) > 3 else None
    
    def extract_text_features(self, row):
        """Extract and combine text features from domain metadata"""
        features = []
        
        # Clean domain
        if 'domain' in row:
            clean_domain = self.clean_domain(row['domain'])
            if clean_domain:
                features.append(clean_domain)
        
        # Add title
        if 'title' in row and row['title'] and str(row['title']).strip() not in ['nan', 'None', '']:
            title = str(row['title']).strip()
            if len(title) > 3:
                features.append(title)
        
        # Add meta description
        if 'meta_description' in row and row['meta_description'] and str(row['meta_description']).strip() not in ['nan', 'None', '']:
            desc = str(row['meta_description']).strip()
            if len(desc) > 10:
                features.append(desc)
        
        combined = ' '.join(features)
        return combined if len(combined.strip()) > 5 else None
    
    def load_and_prepare_data(self, parquet_file):
        """Load parquet file and prepare text data"""
        print(f"📖 Loading data from {parquet_file}...")
        
        # Load parquet file
        df = pd.read_parquet(parquet_file)
        total_rows = len(df)
        print(f"✅ Loaded {total_rows:,} rows")
        
        # Extract text features with progress
        print("📝 Extracting and cleaning text features...")
        texts = []
        valid_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            text = self.extract_text_features(row)
            if text:
                texts.append(text)
                valid_indices.append(idx)
        
        valid_count = len(texts)
        print(f"✅ Extracted {valid_count:,} valid texts ({valid_count/total_rows*100:.1f}%)")
        
        if valid_count < 50:
            raise ValueError(f"Too few valid texts: {valid_count}. Need at least 50.")
        
        return texts, valid_indices, df
    
    def compute_embeddings(self, texts):
        """Compute embeddings with progress tracking"""
        print("🧠 Computing embeddings...")
        
        embeddings = []
        batch_size = 500  # Smaller batches for stability
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="Computing embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                pbar.set_description(f"Batch {i//batch_size + 1}/{total_batches}")
                
                # Compute embeddings for this batch
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_embeddings)
                pbar.update(1)
        
        embeddings_array = np.array(embeddings)
        print(f"✅ Computed embeddings: {embeddings_array.shape}")
        return embeddings_array
    
    def run_bertopic_analysis(self, texts, embeddings):
        """Run BERTopic analysis with progress tracking"""
        print("🎯 Running BERTopic analysis...")
        
        # Create progress tracking for BERTopic phases
        phases = ["UMAP dimensionality reduction", "HDBSCAN clustering", "Topic representation"]
        
        with tqdm(total=len(phases), desc="BERTopic analysis") as pbar:
            pbar.set_description("Starting BERTopic...")
            
            # Run BERTopic fit_transform
            topics, probabilities = self.topic_model.fit_transform(texts, embeddings)
            
            pbar.update(len(phases))  # Complete all phases
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        unique_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
        outliers = list(topics).count(-1)
        
        print(f"✅ BERTopic analysis complete:")
        print(f"   📊 Discovered {unique_topics} topics")
        print(f"   📉 {outliers:,} outliers ({outliers/len(topics)*100:.1f}%)")
        
        return topics, topic_info
    
    def generate_topic_names(self, topic_info):
        """Generate meaningful names for discovered topics"""
        print("🏷️ Generating topic names...")
        
        named_topics = []
        
        for _, row in tqdm(topic_info.iterrows(), total=len(topic_info), desc="Naming topics"):
            topic_id = row['Topic']
            
            if topic_id == -1:  # Skip outliers
                continue
            
            count = row['Count']
            
            # Get topic keywords
            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                keywords = [word for word, _ in topic_words[:8]]
                
                # Create meaningful name from keywords
                name = self.create_topic_name(keywords)
                
                named_topics.append({
                    'topic_id': topic_id,
                    'name': name,
                    'document_count': count,
                    'keywords': keywords[:5],
                    'all_keywords': keywords
                })
        
        # Sort by document count
        named_topics.sort(key=lambda x: x['document_count'], reverse=True)
        
        print(f"✅ Generated names for {len(named_topics)} topics")
        return named_topics
    
    def create_topic_name(self, keywords):
        """Create a meaningful topic name from keywords"""
        if not keywords:
            return "Unknown Topic"
        
        # Clean keywords
        clean_keywords = []
        for kw in keywords[:4]:  # Use top 4 keywords
            if len(kw) > 2 and kw.isalpha():
                clean_keywords.append(kw.title())
        
        if not clean_keywords:
            return "Mixed Content"
        
        # Create name based on keywords
        if len(clean_keywords) == 1:
            return f"{clean_keywords[0]} Related"
        elif len(clean_keywords) == 2:
            return f"{clean_keywords[0]} & {clean_keywords[1]}"
        else:
            return f"{clean_keywords[0]}, {clean_keywords[1]} & Others"
    
    def save_results(self, named_topics, output_dir="domain_categories"):
        """Save results in multiple formats"""
        print("💾 Saving results...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        save_tasks = ["CSV export", "Text summary", "JSON export"]
        
        with tqdm(total=len(save_tasks), desc="Saving results") as pbar:
            # 1. Save CSV
            pbar.set_description("Saving CSV")
            df = pd.DataFrame(named_topics)
            df.to_csv(os.path.join(output_dir, 'discovered_categories.csv'), index=False)
            pbar.update(1)
            
            # 2. Save text summary
            pbar.set_description("Creating summary")
            with open(os.path.join(output_dir, 'categories_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("DISCOVERED DOMAIN CATEGORIES\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Categories: {len(named_topics)}\n\n")
                
                for i, topic in enumerate(named_topics, 1):
                    f.write(f"{i:2d}. {topic['name']}\n")
                    f.write(f"     Documents: {topic['document_count']:,}\n")
                    f.write(f"     Keywords: {', '.join(topic['keywords'])}\n\n")
            pbar.update(1)
            
            # 3. Save JSON
            pbar.set_description("Saving JSON")
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'total_categories': len(named_topics),
                'categories': named_topics
            }
            
            with open(os.path.join(output_dir, 'categories.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            pbar.update(1)
        
        print(f"✅ Results saved to {output_dir}/")
        print(f"   📄 discovered_categories.csv")
        print(f"   📋 categories_summary.txt")
        print(f"   📊 categories.json")
        
        return output_dir
    
    def discover_categories(self, parquet_file):
        """Main method to discover domain categories"""
        start_time = time.time()
        
        try:
            # Phase 1: Setup
            self.setup_models()
            
            # Phase 2: Data loading
            texts, indices, df = self.load_and_prepare_data(parquet_file)
            
            # Phase 3: Compute embeddings
            embeddings = self.compute_embeddings(texts)
            
            # Phase 4: BERTopic analysis
            topics, topic_info = self.run_bertopic_analysis(texts, embeddings)
            
            # Phase 5: Generate names
            named_topics = self.generate_topic_names(topic_info)
            
            # Phase 6: Save results
            output_dir = self.save_results(named_topics)
            
            # Summary
            total_time = time.time() - start_time
            print(f"\n🎉 Category discovery complete!")
            print(f"   📊 {len(named_topics)} categories discovered")
            print(f"   📁 Results saved in: {output_dir}/")
            print(f"   ⏱️ Total time: {total_time/60:.1f} minutes")
            
            return named_topics
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 bertopic_domain_discovery.py <parquet_file>")
        print("Example: python3 bertopic_domain_discovery.py oscar_metadata.parquet")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    
    if not os.path.exists(parquet_file):
        print(f"❌ File not found: {parquet_file}")
        sys.exit(1)
    
    print("🎯 BERTopic Domain Category Discovery")
    print("=" * 50)
    
    # Run discovery
    discovery = DomainCategoryDiscovery()
    try:
        categories = discovery.discover_categories(parquet_file)
        print("\n✅ Discovery completed successfully!")
    except Exception as e:
        print(f"\n❌ Discovery failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Domain and Path Extractor
=========================

This program creates two separate files:
1. Raw domains only (no paths)
2. Domains with specific paths (URLs ending with "/" or "main.php", and truncating at "index.php")

Usage: python create_domain_files.py
"""

import pandas as pd
import sys
import os
import json
from urllib.parse import urlparse
from tqdm import tqdm
import re

def extract_domain(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return None

def process_url_for_paths(url):
    """
    Process URL for path filtering:
    - Keep URLs ending with "/"
    - Keep URLs ending with "main.php"
    - For URLs containing "index.php", truncate at "index.php"
    
    Args:
        url (str): URL to process
        
    Returns:
        str or None: Processed URL if it matches criteria, None otherwise
    """
    try:
        # Check if URL ends with "/"
        if url.endswith('/'):
            return url
        
        # Check if URL ends with "main.php"
        if url.endswith('main.php'):
            return url
        
        # Check if URL contains "index.php" and truncate there
        if 'index.php' in url:
            # Find the position of "index.php"
            index_pos = url.find('index.php')
            if index_pos != -1:
                # Truncate at "index.php" + length of "index.php"
                return url[:index_pos + len('index.php')]
        
        return None
    except:
        return None

def create_domain_files(input_file, domains_file, paths_file):
    """
    Create two files: raw domains and domains with paths
    
    Args:
        input_file (str): Path to input Parquet file
        domains_file (str): Path to output file for raw domains
        paths_file (str): Path to output file for domains with paths
    """
    
    print("🌐 Domain and Path Extractor")
    print("=" * 50)
    print(f"📁 Input file: {input_file}")
    print(f"💾 Raw domains output: {domains_file}")
    print(f"💾 Domains with paths output: {paths_file}")
    print("=" * 50)
    
    # Load data
    print(f"📖 Loading data from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
        print(f"   Loaded {len(df):,} rows with columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Check for URL column
    if 'u' not in df.columns:
        print("❌ Error: 'u' column (URLs) not found in data!")
        return False
    
    print(f"🔄 Processing URLs...")
    
    # Process URLs
    raw_domains = []
    path_urls = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing URLs"):
        url = row['u']
        
        # Extract raw domain
        domain = extract_domain(url)
        if domain:
            raw_domains.append({
                'domain': domain,
                'original_url': url,
                'id': row['id'],
                'collection': row['collection']
            })
        
        # Process for paths
        processed_url = process_url_for_paths(url)
        if processed_url:
            path_urls.append({
                'url': processed_url,
                'original_url': url,
                'id': row['id'],
                'collection': row['collection'],
                'text': row['text']
            })
    
    print(f"\n📊 Processing Results:")
    print(f"   Original URLs: {len(df):,}")
    print(f"   Raw domains extracted: {len(raw_domains):,}")
    print(f"   URLs with matching paths: {len(path_urls):,}")
    
    # Save raw domains file
    print(f"\n💾 Saving raw domains to {domains_file}...")
    try:
        with open(domains_file, 'w', encoding='utf-8') as f:
            for domain_obj in tqdm(raw_domains, desc="Writing domains"):
                f.write(json.dumps(domain_obj, ensure_ascii=False) + '\n')
        print(f"✅ Raw domains saved successfully!")
    except Exception as e:
        print(f"❌ Error saving raw domains: {e}")
        return False
    
    # Save paths file
    print(f"\n💾 Saving URLs with paths to {paths_file}...")
    try:
        with open(paths_file, 'w', encoding='utf-8') as f:
            for path_obj in tqdm(path_urls, desc="Writing paths"):
                f.write(json.dumps(path_obj, ensure_ascii=False) + '\n')
        print(f"✅ URLs with paths saved successfully!")
    except Exception as e:
        print(f"❌ Error saving paths: {e}")
        return False
    
    # Print detailed analysis
    print_analysis(raw_domains, path_urls)
    
    return True

def print_analysis(raw_domains, path_urls):
    """Print detailed analysis of extracted data"""
    print("\n📊 Extraction Analysis:")
    print("=" * 50)
    
    # Raw domains analysis
    print("🌐 Raw Domains Analysis:")
    print("-" * 40)
    
    # Count unique domains
    unique_domains = set(d['domain'] for d in raw_domains)
    print(f"   Total domain entries: {len(raw_domains):,}")
    print(f"   Unique domains: {len(unique_domains):,}")
    print(f"   Avg entries per domain: {len(raw_domains)/len(unique_domains):.1f}")
    
    # Top domains
    from collections import Counter
    domain_counts = Counter(d['domain'] for d in raw_domains)
    print(f"\n   Top 10 Most Common Domains:")
    for domain, count in domain_counts.most_common(10):
        print(f"      {domain:<30}: {count:6,}")
    
    # Path URLs analysis
    print(f"\n🔗 URLs with Paths Analysis:")
    print("-" * 40)
    
    # Analyze path patterns
    slash_endings = sum(1 for p in path_urls if p['url'].endswith('/'))
    main_php_endings = sum(1 for p in path_urls if p['url'].endswith('main.php'))
    index_php_urls = sum(1 for p in path_urls if 'index.php' in p['url'] and not p['url'].endswith('/') and not p['url'].endswith('main.php'))
    
    print(f"   Total URLs with paths: {len(path_urls):,}")
    print(f"   URLs ending with '/': {slash_endings:,} ({slash_endings/len(path_urls)*100:.1f}%)")
    print(f"   URLs ending with 'main.php': {main_php_endings:,} ({main_php_endings/len(path_urls)*100:.1f}%)")
    print(f"   URLs truncated at 'index.php': {index_php_urls:,} ({index_php_urls/len(path_urls)*100:.1f}%)")
    
    # Top domains in paths
    path_domains = [extract_domain(p['url']) for p in path_urls]
    path_domain_counts = Counter(d for d in path_domains if d)
    print(f"\n   Top 10 Domains in Path URLs:")
    for domain, count in path_domain_counts.most_common(10):
        print(f"      {domain:<30}: {count:6,}")
    
    # Sample URLs
    print(f"\n📋 Sample URLs by Pattern:")
    print("-" * 40)
    
    # Sample slash endings
    slash_samples = [p['url'] for p in path_urls if p['url'].endswith('/')][:3]
    if slash_samples:
        print(f"   URLs ending with '/':")
        for i, url in enumerate(slash_samples, 1):
            print(f"      {i}. {url}")
    
    # Sample main.php endings
    main_php_samples = [p['url'] for p in path_urls if p['url'].endswith('main.php')][:3]
    if main_php_samples:
        print(f"   URLs ending with 'main.php':")
        for i, url in enumerate(main_php_samples, 1):
            print(f"      {i}. {url}")
    
    # Sample index.php truncations
    index_php_samples = [p for p in path_urls if 'index.php' in p['url'] and not p['url'].endswith('/') and not p['url'].endswith('main.php')][:3]
    if index_php_samples:
        print(f"   URLs truncated at 'index.php':")
        for i, sample in enumerate(index_php_samples, 1):
            print(f"      {i}. {sample['url']}")
            print(f"         (original: {sample['original_url']})")

def main():
    """Main function"""
    input_file = "filtered_100k_lines.parquet"
    domains_file = "raw_domains.jsonl"
    paths_file = "domains_with_paths.jsonl"
    
    print("🚀 Domain and Path Extractor")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please run the data extraction program first.")
        sys.exit(1)
    
    # Perform extraction
    success = create_domain_files(
        input_file=input_file,
        domains_file=domains_file,
        paths_file=paths_file
    )
    
    if success:
        print(f"\n🎉 Domain and path extraction completed successfully!")
        print(f"📁 Files created:")
        print(f"   - {domains_file}: Raw domains only")
        print(f"   - {paths_file}: Domains with specific paths")
        print(f"\n📊 File contents:")
        print(f"   Raw domains: domain, original_url, id, collection")
        print(f"   Paths: url, original_url, id, collection, text")
        print(f"\n🔍 Path criteria:")
        print(f"   - URLs ending with '/'")
        print(f"   - URLs ending with 'main.php'")
        print(f"   - URLs containing 'index.php' (truncated at index.php)")
    else:
        print(f"\n❌ Domain and path extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
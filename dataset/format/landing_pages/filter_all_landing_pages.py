#!/usr/bin/env python3
"""
All Landing Pages Filter
=========================

This script filters URLs to create a single parquet file containing all landing page domains.
It combines domain-level URLs and landing page paths with comprehensive slash handling.

Landing page types:
1. Domain-level URLs (ending at 3rd slash or less):
   - https://www.wiki.com/
   - https://www.wiki.com (without trailing slash)

2. Landing page paths (with or without trailing slash):
   - https://www.wiki.com/main.php
   - https://www.wiki.com/main.php/
   - https://www.wiki.com/home
   - https://www.wiki.com/home/

Usage: python filter_all_landing_pages.py <input_parquet_file> [--output output_file]

Examples:
  python filter_all_landing_pages.py 4_batch_10000000_100000.parquet
  python filter_all_landing_pages.py data.parquet --output filtered_landing_pages.parquet
"""

import pandas as pd
import sys
import os
import re
import argparse
from urllib.parse import urlparse
from tqdm import tqdm

def is_domain_level_url(url):
    """
    Check if URL is a domain-level URL (stops at third slash with or without trailing slash)
    Examples: 
    - https://www.test.gr/
    - https://www.test.gr
    - https://example.com/
    - https://example.com
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment
        
        # Must have a path that is either empty or just "/"
        # And no query parameters or fragments
        return (path == "" or path == "/") and not query and not fragment
    except:
        return False

def is_landing_page_path(url):
    """
    Check if URL has a landing page path (with or without trailing slash)
    Improved version that handles trailing slashes correctly
    """
    # Define the landing page patterns (without trailing slash)
    landing_files = [
        'home.php', 'main.html', 'main.php', 'default.html', 'default.aspx',
        'landing.html', 'landing.php', 'welcome.html', 'welcome.php',
        'start.html', 'start.php', 'root.html', 'app.html', 'home.aspx',
        'portal.html', 'index.html', 'index.php'
    ]
    
    landing_paths = [
        'home', 'main', 'landing', 'start', 'welcome', 
        'dashboard', 'intro', 'index'
    ]
    
    try:
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment
        
        # No query parameters or fragments allowed
        if query or fragment:
            return False
        
        # Remove trailing slash for comparison
        clean_path = path.rstrip('/')
        
        # Also handle case where path doesn't start with /
        if not clean_path.startswith('/'):
            clean_path = '/' + clean_path
        
        # Check if path exactly matches landing files (with leading slash)
        for landing_file in landing_files:
            if clean_path == f'/{landing_file}':
                return True
        
        # Check if path exactly matches landing paths (with leading slash)
        for landing_path in landing_paths:
            if clean_path == f'/{landing_path}':
                return True
                
        return False
    except:
        return False

def classify_url(url):
    """
    Classify URL and return (is_landing_page, category, reason)
    """
    if is_domain_level_url(url):
        return True, "domain_level", "domain_level_url"
    elif is_landing_page_path(url):
        return True, "landing_path", "landing_page_path"
    else:
        return False, "other", "not_landing_page"

def filter_all_landing_pages(input_file, output_file):
    """
    Filter URLs to create a single parquet file with all landing pages
    
    Args:
        input_file (str): Path to input Parquet file
        output_file (str): Path to output Parquet file
    """
    
    print("🏠 All Landing Pages Filter")
    print("=" * 50)
    print(f"📁 Input file: {input_file}")
    print(f"💾 Output file: {output_file}")
    print("🎯 Filtering for all landing page URLs")
    print("=" * 50)
    
    # Load data
    print(f"📖 Loading data from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        
        # Show data types
        print(f"   Data types:")
        for col in df.columns:
            print(f"     {col}: {df[col].dtype}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Check for URL column
    if 'u' not in df.columns:
        print("❌ Error: 'u' column (URLs) not found in data!")
        return False
    
    # Filter URLs
    print(f"🔄 Analyzing URLs for landing page patterns...")
    
    # Apply landing page filter
    landing_results = []
    categories = []
    reasons = []
    
    for url in tqdm(df['u'], desc="Classifying URLs"):
        is_landing, category, reason = classify_url(url)
        landing_results.append(is_landing)
        categories.append(category)
        reasons.append(reason)
    
    # Create boolean mask
    mask = pd.Series(landing_results)
    
    # Apply filter (keep only original columns)
    filtered_df = df[mask].copy()
    
    print(f"\n📊 Filtering Results:")
    print(f"   Original rows: {len(df):,}")
    print(f"   Landing page rows: {len(filtered_df):,}")
    print(f"   Percentage kept: {len(filtered_df)/len(df)*100:.2f}%")
    
    if len(filtered_df) == 0:
        print("⚠️  No landing page URLs found!")
        return False
    
    # Save results
    print(f"\n💾 Saving filtered results to {output_file}...")
    try:
        filtered_df.to_parquet(output_file, index=False)
        print(f"✅ Results saved successfully!")
        
        # Show output file info
        if os.path.exists(output_file):
            file_size_mb = os.path.getsize(output_file) / (1024 ** 2)
            print(f"   Output file size: {file_size_mb:.2f} MB")
            print(f"   Output columns: {list(filtered_df.columns)}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False
    
    # Print detailed analysis
    print_landing_analysis(filtered_df)
    
    return True

def print_landing_analysis(df):
    """Print detailed analysis of landing page URLs"""
    print("\n📊 Landing Page Analysis:")
    print("=" * 50)
    
    # Classify URLs for analysis
    domain_count = 0
    landing_path_count = 0
    domain_urls = []
    landing_path_urls = []
    
    for url in df['u']:
        if is_domain_level_url(url):
            domain_count += 1
            domain_urls.append(url)
        elif is_landing_page_path(url):
            landing_path_count += 1
            landing_path_urls.append(url)
    
    # Analysis by category
    print("🎯 URLs by Category:")
    print("-" * 30)
    if domain_count > 0:
        percentage = domain_count / len(df) * 100
        print(f"   domain_level        : {domain_count:6,} ({percentage:5.1f}%)")
    if landing_path_count > 0:
        percentage = landing_path_count / len(df) * 100
        print(f"   landing_path        : {landing_path_count:6,} ({percentage:5.1f}%)")
    
    # Analysis by slash count
    print("\n🔗 URLs by Slash Count:")
    print("-" * 30)
    slash_counts = df['u'].apply(lambda x: x.count('/')).value_counts().sort_index()
    for slash_count, count in slash_counts.items():
        percentage = count / len(df) * 100
        print(f"   {slash_count} slashes: {count:6,} ({percentage:5.1f}%)")
    
    # Analysis by domain
    print("\n🌐 Top 10 Domains:")
    print("-" * 30)
    domain_counts = df['u'].apply(lambda x: urlparse(x).netloc).value_counts().head(10)
    for domain, count in domain_counts.items():
        percentage = count / len(df) * 100
        print(f"   {domain:<30}: {count:6,} ({percentage:5.1f}%)")
    
    # Sample URLs by category
    print("\n📋 Sample URLs by Category:")
    print("-" * 30)
    
    if domain_urls:
        print(f"\n   DOMAIN_LEVEL URLs:")
        for i, url in enumerate(domain_urls[:5], 1):
            print(f"      {i}. {url}")
    
    if landing_path_urls:
        print(f"\n   LANDING_PATH URLs:")
        for i, url in enumerate(landing_path_urls[:5], 1):
            print(f"      {i}. {url}")
    
    # Path analysis for landing paths
    if landing_path_urls:
        print("\n🛤️  Landing Path Analysis:")
        print("-" * 30)
        path_counts = {}
        for url in landing_path_urls:
            path = urlparse(url).path
            path_counts[path] = path_counts.get(path, 0) + 1
        
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        for path, count in sorted_paths[:10]:
            percentage = count / len(landing_path_urls) * 100
            print(f"   {path:<20}: {count:6,} ({percentage:5.1f}%)")
    
    # Text content analysis
    if 'text' in df.columns:
        print(f"\n📝 Text Content Statistics:")
        print(f"   Average text length: {df['text'].str.len().mean():.0f} characters")
        print(f"   Empty text entries: {df['text'].str.len().eq(0).sum():,}")
        print(f"   Non-empty text entries: {(df['text'].str.len() > 0).sum():,}")
        
        # Show text_char_count statistics if available
        if 'text_char_count' in df.columns:
            print(f"   Text char count field available: Yes")
            print(f"   Average char count: {df['text_char_count'].mean():.0f}")
            print(f"   Records with text (char count > 0): {(df['text_char_count'] > 0).sum():,}")
        else:
            print(f"   Text char count field available: No")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Filter landing pages from a parquet file')
    parser.add_argument('input_file', help='Path to input parquet file')
    parser.add_argument('--output', '-o', help='Path to output parquet file (optional)')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{base_name}_landing_pages.parquet"
    
    print("🚀 All Landing Pages Filter")
    print("=" * 50)
    print(f"📄 Input file: {args.input_file}")
    print(f"💾 Output file: {args.output}")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        print("Please ensure the input parquet file exists.")
        sys.exit(1)
    
    # Perform filtering
    success = filter_all_landing_pages(
        input_file=args.input_file,
        output_file=args.output
    )
    
    if success:
        print(f"\n🎉 All landing pages filtering completed successfully!")
        print(f"📁 Filtered results saved to: {args.output}")
        print(f"🔍 The parquet file includes:")
        print(f"   - All original columns from input file")
        print(f"   - No additional analysis columns added")
        print(f"\n📊 This file contains all landing page domains:")
        print(f"   🏠 Domain-level URLs (e.g., https://example.com, https://example.com/)")
        print(f"   🎯 Landing page paths (e.g., /main.php, /home, /index.html)")
        print(f"   ✅ Both with and without trailing slashes")
    else:
        print(f"\n❌ All landing pages filtering failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
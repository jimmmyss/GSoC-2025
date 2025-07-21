#!/usr/bin/env python3
"""
Domain Categorizer using Gemini API
===================================

This script accesses actual websites and categorizes them using Gemini AI
with Greek categories. Each domain is processed individually and saved 
immediately to the output file.

Usage: python categorize_domains_gemini.py <parquet_file>
"""

import pandas as pd
import google.generativeai as genai
import sys
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import gc
import time
import re
from urllib.parse import urljoin, urlparse
import warnings
warnings.filterwarnings('ignore')

# Configure Gemini API
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

# Greek categories
CATEGORIES = [
    "Ηλεκτρονικό Εμπόριο & Αγορές",
    "Ειδήσεις & Μέσα Ενημέρωσης",
    "Κοινωνικά Δίκτυα & Κοινότητα",
    "Τεχνολογία & Λογισμικό",
    "Ψυχαγωγία & Μέσα",
    "Εκπαίδευση & Έρευνα",
    "Υγεία & Ιατρική",
    "Κυβέρνηση & Δημόσιες Υπηρεσίες",
    "Ταξίδια & Τουρισμός",
    "Χρηματοοικονομικά & Τραπεζικά",
    "Αθλητισμός & Αναψυχή",
    "Άλλο"
]

# Individual domain processing - no batching

# Request settings
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def setup_gemini():
    """Setup Gemini model"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        print(f"Error setting up Gemini model: {e}")
        sys.exit(1)

def clean_domain(domain):
    """Clean and normalize domain name"""
    if not domain:
        return None
    
    domain = str(domain).strip().lower()
    
    # Remove protocol if present
    if domain.startswith('http://'):
        domain = domain[7:]
    elif domain.startswith('https://'):
        domain = domain[8:]
    
    # Remove www if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Remove trailing slash and path
    if '/' in domain:
        domain = domain.split('/')[0]
    
    # Basic validation
    if not domain or '.' not in domain:
        return None
    
    return domain

def fetch_webpage_content(domain):
    """Fetch and extract content from a webpage"""
    if not domain:
        return None, "Invalid domain"
    
    # Try both http and https
    urls_to_try = [
        f"https://{domain}",
        f"http://{domain}",
        f"https://www.{domain}",
        f"http://www.{domain}"
    ]
    
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    for url in urls_to_try:
        try:
            response = requests.get(
                url, 
                headers=headers, 
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=False
            )
            
            if response.status_code == 200:
                return extract_content_from_html(response.text, url), None
            
        except requests.exceptions.RequestException as e:
            continue
    
    return None, "Failed to fetch webpage"

def extract_content_from_html(html_content, url):
    """Extract meaningful content from HTML"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        content_parts = []
        
        # Extract title
        title = soup.find('title')
        if title:
            content_parts.append(f"Title: {title.get_text().strip()}")
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            content_parts.append(f"Description: {meta_desc.get('content', '').strip()}")
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            content_parts.append(f"Keywords: {meta_keywords.get('content', '').strip()}")
        
        # Extract Open Graph data
        og_title = soup.find('meta', property='og:title')
        if og_title:
            content_parts.append(f"OG Title: {og_title.get('content', '').strip()}")
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            content_parts.append(f"OG Description: {og_desc.get('content', '').strip()}")
        
        # Extract main content
        main_content = ""
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '#content', 
            '.main', '#main', '.post', '.entry-content'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                main_content = main_element.get_text(separator=' ', strip=True)
                break
        
        # If no main content found, get text from body
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)
        
        # Clean and truncate main content
        if main_content:
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            main_content = main_content[:1000]  # Limit to 1000 chars
            content_parts.append(f"Content: {main_content}")
        
        return " | ".join(content_parts) if content_parts else None
        
    except Exception as e:
        return None

def create_categorization_prompt(content, categories_list):
    """Create a prompt for Gemini to categorize the website content"""
    categories_formatted = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories_list)])
    
    prompt = f"""Είσαι ειδικός σε κατηγοριοποίηση ιστότοπων. Με βάση τις παρακάτω πληροφορίες ιστότοπου, κατηγοριοποίησέ τον σε ΜΙΑ από τις παρακάτω κατηγορίες:

{categories_formatted}

Πληροφορίες Ιστότοπου:
{content}

Οδηγίες:
- Επέστρεψε ΜΟΝΟ το όνομα της κατηγορίας ακριβώς όπως παρατίθεται παραπάνω
- Διάλεξε την πιο κατάλληλη κατηγορία με βάση τον κύριο σκοπό του ιστότοπου
- Αν ο ιστότοπος δεν ταιριάζει σαφώς σε καμία κατηγορία, διάλεξε "Άλλο"
- Μη δώσεις εξηγήσεις, μόνο το όνομα της κατηγορίας

Κατηγορία:"""
    
    return prompt

def query_gemini_with_retry(model, prompt, max_retries=3):
    """Query Gemini with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                wait_time = (2 ** attempt) * 5  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
    return None

def categorize_domain(domain, model):
    """Categorize domain by fetching its content and using Gemini"""
    # Clean domain
    cleaned_domain = clean_domain(domain)
    if not cleaned_domain:
        return "Άλλο", 0.1, "Invalid domain"
    
    # Fetch webpage content
    content, error = fetch_webpage_content(cleaned_domain)
    if not content:
        return "Άλλο", 0.1, error or "Failed to fetch content"
    
    # Truncate content if too long
    if len(content) > 3000:
        content = content[:3000] + "..."
    
    # Get category using Gemini
    prompt = create_categorization_prompt(content, CATEGORIES)
    response = query_gemini_with_retry(model, prompt)
    
    category = "Άλλο"  # Default
    confidence = 0.5   # Default confidence
    
    if response:
        # Clean the response and find matching category
        clean_response = response.strip().strip('"').strip("'")
        for cat in CATEGORIES:
            if cat in clean_response or clean_response in cat:
                category = cat
                confidence = 0.8  # High confidence if found
                break
    
    return category, confidence, "Success"

def find_last_processed_index(output_file, input_df):
    """Find the last processed index by reading existing output file"""
    if not os.path.exists(output_file):
        print("📋 No existing output file found - starting fresh")
        return 0
    
    try:
        existing_df = pd.read_parquet(output_file)
        processed_count = len(existing_df)
        
        if processed_count == 0:
            print("📋 Output file exists but is empty - starting fresh")
            return 0
        
        # Validate that we can resume properly
        if processed_count >= len(input_df):
            print(f"✅ All {len(input_df)} domains already processed!")
            return len(input_df)
        
        print(f"📋 Found existing output with {processed_count:,} processed domains")
        print(f"🔄 Will resume from domain {processed_count + 1}")
        
        # Show last few processed domains for verification
        if processed_count > 0:
            last_domains = existing_df.tail(3)
            print("📝 Last processed domains:")
            for _, row in last_domains.iterrows():
                domain_col = None
                # Try to find domain column
                for col in ['domain', 'Domain', 'url', 'URL', 'website', 'site']:
                    if col in row.index:
                        domain_col = col
                        break
                if not domain_col and len(row.index) > 0:
                    domain_col = row.index[0]
                
                if domain_col:
                    domain = str(row[domain_col])[:30]
                    category = str(row.get('category', 'N/A'))[:20]
                    print(f"   {domain} → {category}")
        
        return processed_count
        
    except Exception as e:
        print(f"⚠️  Warning: Could not read existing output file ({e})")
        print("📋 Starting fresh - will overwrite existing file")
        return 0

def save_single_result(result_row, output_file):
    """Save single domain result to parquet file with error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(output_file):
                # Append to existing file
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, result_row], ignore_index=True)
                combined_df.to_parquet(output_file, index=False)
            else:
                # Create new file
                result_row.to_parquet(output_file, index=False)
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Save attempt {attempt + 1} failed: {e}")
                print(f"🔄 Retrying in 1 second...")
                time.sleep(1)
            else:
                print(f"❌ Failed to save result after {max_retries} attempts: {e}")
                return False
    
    return False

def analyze_input_file(df):
    """Analyze input file and find domain column"""
    print("Analyzing input file...")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Try to find domain column
    domain_column = None
    possible_domain_columns = ['domain', 'Domain', 'url', 'URL', 'website', 'site']
    
    for col in possible_domain_columns:
        if col in df.columns:
            domain_column = col
            break
    
    if not domain_column and len(df.columns) > 0:
        domain_column = df.columns[0]  # Use first column as fallback
    
    if domain_column:
        print(f"Using column '{domain_column}' as domain source")
        sample_domains = df[domain_column].dropna().head(5).tolist()
        print(f"Sample domains: {sample_domains}")
    else:
        print("ERROR: Could not find domain column!")
        sys.exit(1)
    
    return domain_column

def main():
    """Main function"""
    # Check for help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("🔧 Domain Categorizer with Gemini AI")
        print("=" * 50)
        print("Usage: python categorize_domains_gemini.py <parquet_file> [options]")
        print("\nOptions:")
        print("  --restart, --fresh  Force restart from beginning (ignore existing output)")
        print("  --help, -h          Show this help message")
        print("\nExamples:")
        print("  python categorize_domains_gemini.py my_domains.parquet")
        print("  python categorize_domains_gemini.py my_domains.parquet --restart")
        print("\nResume capability:")
        print("  - Automatically resumes by reading existing output file")
        print("  - Use --restart to force start from beginning")
        print("  - Each domain is saved immediately upon processing")
        print("  - No backup files created - uses existing output file only")
        sys.exit(0)
    
    # Parse arguments (filter out options)
    parquet_files = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if len(parquet_files) != 1:
        print("Usage: python categorize_domains_gemini.py <parquet_file> [options]")
        print("Use --help for more information")
        sys.exit(1)
    
    input_file = parquet_files[0]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_categorized_domains.parquet"
    
    # Setup Gemini model
    print("Setting up Gemini model...")
    model = setup_gemini()
    print("Gemini model ready!")
    
    # Read input file
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Analyze input file
    domain_column = analyze_input_file(df)
    
    # Display categories
    print("\nCategories:")
    for i, category in enumerate(CATEGORIES, 1):
        print(f"  {i}. {category}")
    print()
    
    # Check for command line options
    force_restart = "--restart" in sys.argv or "--fresh" in sys.argv
    
    # Find last processed index
    total_rows = len(df)
    
    if force_restart:
        print("🔄 Force restart requested - starting from beginning")
        start_idx = 0
    else:
        start_idx = find_last_processed_index(output_file, df)
    
    # Handle resuming
    if start_idx >= total_rows:
        print("🎉 All domains already processed - nothing to do!")
        print(f"📊 Output file: {output_file}")
        return
    elif start_idx > 0:
        print(f"🔄 Resuming processing from domain {start_idx + 1} of {total_rows}")
        df = df.iloc[start_idx:].reset_index(drop=True)
    else:
        print("🚀 Starting fresh processing")
    
    remaining_rows = len(df)
    print(f"Processing {remaining_rows} domains individually...")
    
    # Process each domain individually
    processed_count = 0
    failed_saves = 0
    
    try:
        with tqdm(total=remaining_rows, desc="Processing domains") as pbar:
            for idx, row in df.iterrows():
                domain = row[domain_column]
                current_position = start_idx + idx + 1
                
                # Categorize domain
                category, confidence, status = categorize_domain(domain, model)
                
                # Create result DataFrame for this domain
                result_row = pd.DataFrame([row]).copy()
                result_row['category'] = category
                result_row['confidence'] = confidence
                result_row['status'] = status
                
                # Save result immediately with error handling
                save_success = save_single_result(result_row, output_file)
                if save_success:
                    processed_count += 1
                else:
                    failed_saves += 1
                    print(f"⚠️  Skipping save for domain {domain} due to save errors")
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'domain': str(domain)[:20] + ('...' if len(str(domain)) > 20 else ''),
                    'category': category[:15] + ('...' if len(category) > 15 else ''),
                    'confidence': f"{confidence:.2f}",
                    'position': f"{current_position}/{total_rows}"
                })
                
                # Delay to respect rate limits
                time.sleep(0.5)
                
                # Periodic memory cleanup
                if (idx + 1) % 50 == 0:
                    gc.collect()
                    
                # Progress checkpoint every 100 domains
                if (idx + 1) % 100 == 0:
                    print(f"\n✅ Checkpoint: {processed_count} domains processed successfully")
                    if failed_saves > 0:
                        print(f"⚠️  {failed_saves} save failures encountered")
                
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Process interrupted by user")
        print(f"✅ Progress saved to: {output_file}")
        print(f"📊 Successfully processed: {processed_count} domains")
        if failed_saves > 0:
            print(f"⚠️  Save failures: {failed_saves} domains")
        print(f"\n🔄 To resume processing, run the same command again:")
        print(f"   python categorize_domains_gemini.py {sys.argv[1]}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print(f"✅ Progress saved to: {output_file}")
        print(f"📊 Successfully processed: {processed_count} domains") 
        if failed_saves > 0:
            print(f"⚠️  Save failures: {failed_saves} domains")
        print(f"\n🔄 To resume processing, run the same command again:")
        print(f"   python categorize_domains_gemini.py {sys.argv[1]}")
        sys.exit(1)
    
    # Final summary
    print(f"\n🎉 Processing complete! Results saved to: {output_file}")
    print(f"✅ Successfully processed: {processed_count} domains")
    if failed_saves > 0:
        print(f"⚠️  Save failures: {failed_saves} domains (check logs above)")
    
    # Show summary statistics
    try:
        final_df = pd.read_parquet(output_file)
        print(f"\nSummary Statistics:")
        print(f"Total domains processed: {len(final_df):,}")
        
        print("\nCategories:")
        category_counts = final_df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nAverage confidence: {final_df['confidence'].mean():.3f}")
        
        print("\nStatus Summary:")
        status_counts = final_df['status'].value_counts()
        for status, count in status_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"  {status}: {count:,} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Could not generate summary: {e}")

if __name__ == "__main__":
    main() 
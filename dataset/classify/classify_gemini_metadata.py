import pandas as pd
import google.generativeai as genai
import sys
import os
from tqdm import tqdm
import gc
import json
import time
import re

# Configure Gemini API
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

# Define categories and subcategories
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

# Set batch size
BATCH_SIZE = 100  # Reduced for API rate limiting

def setup_gemini():
    """Setup Gemini model"""
    try:
        # Using Gemini 2.5 Flash - best balance of performance, speed, and cost
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        print(f"Error setting up Gemini model: {e}")
        sys.exit(1)

def get_text_for_categorization(row, df_columns):
    """Extract text from row for categorization using Oscar metadata fields"""
    parts = []
    extracted_fields = {}
    
    # Define the fields we want to use for categorization
    target_fields = [
        'domain', 'redirect_url', 'title', 'meta_description',
        'keywords', 'og_title', 'og_description', 'og_type'
    ]
    
    # Try to find each field by name, with fallback to common variations
    field_variations = {
        'domain': ['domain', 'Domain'],
        'redirect_url': ['redirect_url', 'redirect_URL', 'url', 'URL'],
        'title': ['title', 'Title'],
        'meta_description': ['meta_description', 'meta_desc', 'description', 'Description'],
        'keywords': ['keywords', 'Keywords'],
        'og_title': ['og_title', 'og:title', 'og_Title'],
        'og_description': ['og_description', 'og:description', 'og_desc'],
        'og_type': ['og_type', 'og:type', 'og_Type']
    }
    
    # Extract text from each available field
    for field in target_fields:
        field_value = None
        
        # Try to find the field by checking variations
        for variation in field_variations.get(field, [field]):
            if variation in df_columns:
                field_value = row[variation]
                break
        
        # If not found by name, try positional fallback for common fields
        if field_value is None:
            if field == 'domain' and len(df_columns) >= 1:
                field_value = row.iloc[0]
            elif field == 'title' and len(df_columns) >= 3:
                field_value = row.iloc[2]
            elif field == 'meta_description' and len(df_columns) >= 4:
                field_value = row.iloc[3]
        
        # Store field value for later checking
        if pd.notna(field_value) and str(field_value).strip():
            cleaned_value = str(field_value).strip()
            if cleaned_value and cleaned_value.lower() not in ['none', 'null', 'nan']:
                extracted_fields[field] = cleaned_value
                parts.append(cleaned_value)
    
    combined_text = ' '.join(parts) if parts else None
    return combined_text, extracted_fields



def create_categorization_prompt(text, categories_list):
    """Create a prompt for Gemini to categorize the text"""
    categories_formatted = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories_list)])
    
    prompt = f"""You are a website categorization expert. Based on the following website information, classify it into ONE of these categories:

{categories_formatted}

Website Information:
{text}

Instructions:
- Return ONLY the category name exactly as listed above
- Choose the most appropriate category based on the website's main purpose
- If the website doesn't clearly fit any category, choose "Other"
- Do not provide explanations, just the category name

Category:"""
    
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

def categorize_text(text, extracted_fields, model):
    """Categorize text using Gemini - main category only"""
    if not text:
        return "Other", 1.0
    
    # Truncate text if too long
    text = text[:2000] if len(text) > 2000 else text
    
    # Get main category
    main_prompt = create_categorization_prompt(text, CATEGORIES)
    main_response = query_gemini_with_retry(model, main_prompt)
    
    main_category = "Other"  # Default
    main_score = 0.5  # Default confidence
    
    if main_response:
        # Clean the response and find matching category
        clean_response = main_response.strip().strip('"').strip("'")
        for category in CATEGORIES:
            if category.lower() in clean_response.lower() or clean_response.lower() in category.lower():
                main_category = category
                main_score = 0.8  # High confidence if found
                break
    
    return main_category, main_score

def find_last_completed_batch(output_file):
    """Find the last completed batch in the output file"""
    if not os.path.exists(output_file):
        return 0
    
    try:
        df = pd.read_parquet(output_file)
        # Check if all rows in the last batch are complete
        last_batch_start = (len(df) // BATCH_SIZE) * BATCH_SIZE
        if last_batch_start == len(df):
            return last_batch_start
        return last_batch_start - BATCH_SIZE
    except:
        return 0

def process_batch(df, start_idx, end_idx, model, progress_bar):
    """Process a batch of rows"""
    results = []
    df_columns = df.columns
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        text, extracted_fields = get_text_for_categorization(row, df_columns)
        main_category, main_score = categorize_text(
            text, extracted_fields, model
        )
        results.append((main_category, main_score))
        progress_bar.update(1)
        
        # Add small delay to respect API rate limits
        time.sleep(0.1)
    
    return results

def clear_memory():
    """Clear memory after each batch"""
    gc.collect()

def save_batch(batch_df, output_file, is_first_batch):
    """Save a batch to the parquet file, either creating new or appending"""
    if is_first_batch:
        batch_df.to_parquet(output_file)
    else:
        # Read existing data
        existing_df = pd.read_parquet(output_file)
        # Concatenate with new batch
        combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
        # Save back to file
        combined_df.to_parquet(output_file)

def analyze_dataframe_columns(df):
    """Analyze and report which Oscar metadata fields are available"""
    target_fields = [
        'domain', 'redirect_url', 'title', 'meta_description',
        'keywords', 'og_title', 'og_description', 'og_type'
    ]
    
    field_variations = {
        'domain': ['domain', 'Domain'],
        'redirect_url': ['redirect_url', 'redirect_URL', 'url', 'URL'],
        'title': ['title', 'Title'],
        'meta_description': ['meta_description', 'meta_desc', 'description', 'Description'],
        'keywords': ['keywords', 'Keywords'],
        'og_title': ['og_title', 'og:title', 'og_Title'],
        'og_description': ['og_description', 'og:description', 'og_desc'],
        'og_type': ['og_type', 'og:type', 'og_Type']
    }
    
    available_fields = []
    missing_fields = []
    
    print("Analyzing Oscar metadata fields:")
    print(f"Total columns in dataset: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print()
    
    for field in target_fields:
        found = False
        found_as = None
        
        # Check if field exists by name
        for variation in field_variations.get(field, [field]):
            if variation in df.columns:
                available_fields.append(field)
                found_as = variation
                found = True
                break
        
        # Check positional fallback
        if not found:
            if field == 'domain' and len(df.columns) >= 1:
                available_fields.append(field)
                found_as = f"position 0 ({df.columns[0]})"
                found = True
            elif field == 'title' and len(df.columns) >= 3:
                available_fields.append(field)
                found_as = f"position 2 ({df.columns[2]})"
                found = True
            elif field == 'meta_description' and len(df.columns) >= 4:
                available_fields.append(field)
                found_as = f"position 3 ({df.columns[3]})"
                found = True
        
        if found:
            print(f"✓ {field}: found as '{found_as}'")
        else:
            missing_fields.append(field)
            print(f"✗ {field}: not found")
    
    print(f"\nUsing {len(available_fields)} out of {len(target_fields)} Oscar metadata fields for categorization")
    if missing_fields:
        print(f"Missing fields: {', '.join(missing_fields)}")
    print()
    
    return available_fields

def main():
    # Check if input file is provided
    if len(sys.argv) != 2:
        print("Usage: python categorize_parquet_gemini.py <input_parquet_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_categorized_gemini.parquet"
    
    # Setup Gemini model
    print("Setting up Gemini model...")
    model = setup_gemini()
    print("Gemini model ready!")
    
    # Read the input file
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    total_rows = len(df)
    
    # Analyze available Oscar metadata fields
    available_fields = analyze_dataframe_columns(df)
    
    # Display categories
    print("Categories:")
    for i, category in enumerate(CATEGORIES, 1):
        print(f"  {i}. {category}")
    print()
    
    # Find the last completed batch
    start_idx = find_last_completed_batch(output_file)
    if start_idx > 0:
        print(f"Resuming from batch {start_idx//BATCH_SIZE + 1} (starting at row {start_idx})")
    else:
        print("Starting from the beginning")
    
    print(f"\nProcessing {total_rows} rows in batches of {BATCH_SIZE}...")
    print("Note: Using Gemini API - main categories only (no subcategories)")
    
    # Create progress bar for overall progress
    with tqdm(total=total_rows, desc="Overall Progress", position=0) as pbar:
        try:
            for batch_start in range(start_idx, total_rows, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_rows)
                batch_size = batch_end - batch_start
                
                print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1} (rows {batch_start} to {batch_end-1})...")
                
                # Create progress bar for current batch
                with tqdm(total=batch_size, desc=f"Batch {batch_start//BATCH_SIZE + 1}", position=1, leave=False) as batch_pbar:
                    # Process batch
                    results = process_batch(df, batch_start, batch_end, model, batch_pbar)
                    
                    # Create a new DataFrame for this batch
                    batch_df = df.iloc[batch_start:batch_end].copy()
                    batch_df['main_category'] = None
                    batch_df['main_confidence_score'] = None
                    
                    # Update batch DataFrame with results
                    for i, (main_category, main_score) in enumerate(results):
                        batch_df.iloc[i, batch_df.columns.get_loc('main_category')] = main_category
                        batch_df.iloc[i, batch_df.columns.get_loc('main_confidence_score')] = main_score
                    
                    # Save this batch
                    is_first_batch = not os.path.exists(output_file)
                    save_batch(batch_df, output_file, is_first_batch)
                    
                    # Update overall progress
                    pbar.update(batch_size)
                
                print(f"Saved batch {batch_start//BATCH_SIZE + 1} ({batch_end/total_rows*100:.1f}% complete)")
                
                # Clear memory after each batch
                clear_memory()
                
        except KeyboardInterrupt:
            print("\nProcess interrupted. Saving progress...")
            print(f"Progress saved. You can resume from batch {batch_start//BATCH_SIZE + 1} (starting at row {batch_start})")
            sys.exit(1)
    
    print(f"\nProcessing complete! Results saved to {output_file}")
    
    # Show summary statistics
    final_df = pd.read_parquet(output_file)
    print("\nCategorization Summary:")
    print("Main Categories:")
    main_cat_counts = final_df['main_category'].value_counts()
    for cat, count in main_cat_counts.items():
        print(f"  {cat}: {count} ({count/len(final_df)*100:.1f}%)")
    
    print(f"\nTotal entries categorized: {len(final_df)}")
    print(f"Average confidence score: {final_df['main_confidence_score'].mean():.3f}")

if __name__ == "__main__":
    main()

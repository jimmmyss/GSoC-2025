import os
import re
import unicodedata
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import pytesseract
from tqdm import tqdm

# Initialize Hunspell dictionaries for English and Greek
try:
    import hunspell
    hunspell_en = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    hunspell_el = hunspell.HunSpell('/usr/share/hunspell/el_GR.dic', '/usr/share/hunspell/el_GR.aff')
    print("Hunspell dictionaries initialized successfully")
except ImportError:
    print("Warning: hunspell module not installed. Install with: sudo apt install python3-hunspell")
    print("Spell checking will be disabled - only accent stripping and character normalization will be performed.")
    hunspell_en = None
    hunspell_el = None
except Exception as e:
    print(f"Warning: Could not initialize Hunspell dictionaries: {e}")
    print("Spell checking will be disabled - only accent stripping and character normalization will be performed.")
    hunspell_en = None
    hunspell_el = None

def remove_greek_accents(text):
    """Remove ALL Greek accents, diacritics, and combining characters using Unicode ranges only"""
    # First decompose the text to separate base characters and diacritics
    decomposed = unicodedata.normalize('NFD', text)
    
    # Define ALL possible diacritics and special character ranges to remove
    diacritics_ranges = [
        (0x0300, 0x036F),  # Combining diacritical marks
        (0x1DC0, 0x1DFF),  # Combining diacritical marks supplement
        (0x20D0, 0x20FF),  # Combining diacritical marks for symbols
        (0x0342, 0x0345),  # Greek specific combining marks
        (0x0360, 0x0361),  # Additional combining marks
        (0x0483, 0x0489),  # Cyrillic combining marks
        (0xFE20, 0xFE2F),  # Combining half marks
        (0x0344, 0x0344),  # Combining Greek dialytika tonos
        (0x0340, 0x0341),  # Combining grave tone mark and acute tone mark
        (0x1AB0, 0x1AFF),  # Combining diacritical marks extended
        (0x1F00, 0x1FFE),  # Greek Extended (includes all polytonic combinations)
    ]
    
    # Create a set of all diacritical marks for faster lookup
    diacritical_chars = {chr(i) for start, end in diacritics_ranges 
                        for i in range(start, end + 1)}
    
    # Remove combining characters and normalize precomposed accented characters
    result = ''.join(c for c in decomposed 
                    if not unicodedata.combining(c) and c not in diacritical_chars)
    
    # Additional pass to handle precomposed Greek characters by converting to base form
    # Use Unicode normalization to convert precomposed to decomposed, then filter
    final_decomposed = unicodedata.normalize('NFD', result)
    final = ''.join(c for c in final_decomposed 
                   if not unicodedata.combining(c) and c not in diacritical_chars)
    
    return final

def preprocess_image(image):
    """
    Preprocess the image for better OCR accuracy:
    - Convert to grayscale
    - Resize (double resolution)
    - Sharpen
    """
    gray = image.convert('L')
    resized = gray.resize((gray.width * 2, gray.height * 2), resample=Image.Resampling.LANCZOS)
    sharpened = resized.filter(ImageFilter.SHARPEN)
    return sharpened

def is_greek_text(word):
    """Check if word contains primarily Greek characters"""
    greek_chars = sum(1 for c in word if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
    total_chars = sum(1 for c in word if c.isalpha())
    return total_chars > 0 and greek_chars / total_chars > 0.5

def safe_encode_greek(word):
    """Safely encode Greek word to ISO-8859-7 for Hunspell"""
    try:
        # Use ISO-8859-7 encoding for Greek as required by Hunspell
        return word.encode('iso-8859-7')
    except UnicodeEncodeError:
        # Fallback to UTF-8 if ISO-8859-7 fails
        try:
            return word.encode('utf-8')
        except UnicodeEncodeError:
            return None

def safe_decode_greek(byte_word):
    """Safely decode Greek word from ISO-8859-7"""
    try:
        if isinstance(byte_word, bytes):
            # Try ISO-8859-7 first, then fallback to UTF-8
            try:
                return byte_word.decode('iso-8859-7')
            except UnicodeDecodeError:
                return byte_word.decode('utf-8')
        return byte_word
    except UnicodeDecodeError:
        return None

# Global set to track words not found in dictionary
words_not_in_dictionary = set()

def normalize_greek_characters(text):
    """Normalize problematic characters that OCR might confuse"""
    # Convert micro sign (U+00B5) to Greek small letter mu (U+03BC)
    text = text.replace('\u00B5', '\u03BC')  # µ → μ
    return text

def correct_word_with_accents(word):
    """Get Hunspell suggestion for every word and replace with first suggestion (accented version)"""
    global words_not_in_dictionary
    norm_word = unicodedata.normalize('NFC', word)
    
    if not norm_word.isalpha():
        return word  # Preserve punctuation, digits, etc.

    # If Hunspell is not available, just return the word
    if hunspell_en is None or hunspell_el is None:
        return norm_word

    # Check if the word is primarily Greek
    if is_greek_text(norm_word):
        try:
            # Check if the word is already spelled correctly
            is_correct = hunspell_el.spell(norm_word)
            
            if is_correct:
                # Word is already correct, return it as-is
                return norm_word
            else:
                # Word is incorrect, get suggestions
                suggestions = hunspell_el.suggest(norm_word)
                
                if suggestions:
                    # Use the first suggestion (which should be the properly accented version)
                    best_suggestion = suggestions[0]
                    
                    # Handle the encoding issue: hunspell returns strings with ISO-8859-7 chars as Latin-1
                    if isinstance(best_suggestion, str):
                        try:
                            # The string contains ISO-8859-7 bytes interpreted as Latin-1
                            # First encode as Latin-1 to get the original bytes, then decode as ISO-8859-7
                            iso_bytes = best_suggestion.encode('latin-1')
                            final_suggestion = iso_bytes.decode('iso-8859-7')
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            # If conversion fails, use the original string
                            final_suggestion = best_suggestion
                    elif isinstance(best_suggestion, bytes):
                        try:
                            # If it's bytes, try ISO-8859-7 first, then UTF-8
                            final_suggestion = best_suggestion.decode('iso-8859-7')
                        except UnicodeDecodeError:
                            try:
                                final_suggestion = best_suggestion.decode('utf-8')
                            except UnicodeDecodeError:
                                final_suggestion = best_suggestion.decode('utf-8', errors='ignore')
                    else:
                        final_suggestion = str(best_suggestion)
                    
                    # Normalize to UTF-8
                    return unicodedata.normalize('NFC', final_suggestion)
                else:
                    # No suggestions available
                    words_not_in_dictionary.add(norm_word)
                    return norm_word
            
        except Exception as e:
            print(f"Error processing Greek word '{norm_word}': {e}")
            words_not_in_dictionary.add(norm_word)
            return norm_word
    
    # For English words - check spelling first, then get suggestions if needed
    try:
        is_correct = hunspell_en.spell(norm_word)
        
        if is_correct:
            # Word is already correct, return it as-is
            return norm_word
        else:
            # Word is incorrect, get suggestions
            suggestions = hunspell_en.suggest(norm_word)
            if suggestions:
                best_suggestion = suggestions[0]
                # Ensure proper UTF-8 encoding for English words too
                if isinstance(best_suggestion, bytes):
                    final_suggestion = best_suggestion.decode('utf-8', errors='ignore')
                else:
                    final_suggestion = best_suggestion
                return unicodedata.normalize('NFC', final_suggestion)
            else:
                # No suggestions available, add to unknown words
                words_not_in_dictionary.add(norm_word)
                return norm_word
        
    except Exception as e:
        print(f"Error processing English word '{norm_word}': {e}")
        words_not_in_dictionary.add(norm_word)
        return norm_word

def correct_text_spelling_preserve_format(text):
    """Apply spell correction while preserving original formatting"""
    # Text should already be normalized in the stripped version
    if hunspell_en is None or hunspell_el is None:
        print("Hunspell not available - returning text without accent restoration")
        return text
    
    print("Processing text with UTF-8 encoding (hunspell handles internal conversion)")
    iso_text = text
    
    def replace_word_with_progress(match):
        word = match.group(0)
        corrected = correct_word_with_accents(word)
        progress_counter['count'] += 1
        
        # Update progress bar every 100 words to avoid too frequent updates
        if progress_counter['count'] % 100 == 0 or progress_counter['count'] == total_words:
            progress_bar.update(100 if progress_counter['count'] % 100 == 0 else total_words % 100)
        
        return corrected
    
    # Use regex to find words (sequences of letters, including Greek)
    word_pattern = r'\b[\w\u0370-\u03FF\u1F00-\u1FFF]+\b'
    
    # Find all words first to get total count for progress bar
    words = re.findall(word_pattern, iso_text)
    total_words = len(words)
    
    # Create a progress tracker
    progress_counter = {'count': 0}
    
    # Initialize progress bar
    with tqdm(total=total_words, desc="Spell checking", unit="words") as progress_bar:
        corrected_text = re.sub(word_pattern, replace_word_with_progress, iso_text)
    
    # Ensure the final result is in UTF-8
    try:
        # Normalize to UTF-8 for final output
        final_text = unicodedata.normalize('NFC', corrected_text)
        print("Successfully converted corrected text back to UTF-8")
        return final_text
    except Exception as e:
        print(f"Warning: Error normalizing final text: {e}")
        return corrected_text

def create_stripped_text(text):
    """Create version with Greek accents removed and characters normalized"""
    # First normalize problematic OCR characters (µ → μ)
    normalized_text = normalize_greek_characters(text)
    # Then remove accents
    return remove_greek_accents(normalized_text)

def ocr_pdf(pdf_path, dpi=300, lang='ell+eng', save_output=False, output_path=None):
    """
    Convert PDF pages to images, preprocess, OCR (Greek + English),
    save raw OCR text, stripped text, and spell-checked text with proper accents.

    Args:
        pdf_path (str): Path to input PDF
        dpi (int): DPI for pdf2image conversion
        lang (str): Tesseract languages (default 'ell+eng')
        save_output (bool): Save output files
        output_path (str): Base path to save files; if None, derived from pdf_path

    Returns:
        str: Spell-corrected OCR text
    """
    global words_not_in_dictionary
    words_not_in_dictionary = set()  # Reset the set for each PDF
    
    print(f"Processing: {pdf_path}")

    images = convert_from_path(pdf_path, dpi=dpi)
    all_text = []

    for i, img in enumerate(tqdm(images, desc="Pages OCR")):
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang=lang, config='--psm 6')
        all_text.append(f"\n--- Page {i + 1} ---\n{text}")

    full_text = "".join(all_text)  # preserve original spacing, including trailing spaces/newlines

    if save_output:
        base_out_path = output_path or os.path.splitext(pdf_path)[0]
        
        # 1. Save raw OCR output
        raw_output_file = base_out_path + "_ocr.txt"
        with open(raw_output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Raw OCR output saved to: {raw_output_file}")

    # 2. Create stripped version (normalize characters and remove accents)
    print("Normalizing characters (µ → μ) and stripping accents from OCR output...")
    stripped_text = create_stripped_text(full_text)
    
    if save_output:
        stripped_output_file = base_out_path + "_ocr_stripped.txt"
        with open(stripped_output_file, 'w', encoding='utf-8') as f:
            f.write(stripped_text)
        print(f"Stripped OCR output saved to: {stripped_output_file}")

    # 3. Pass EVERY word through Hunspell to get accented suggestions
    print("Processing through Hunspell to restore accents...")
    corrected_text = correct_text_spelling_preserve_format(stripped_text)

    if save_output:
        base_out_path = output_path or os.path.splitext(pdf_path)[0]
        corrected_output_file = base_out_path + "_ocr_corrected.txt"
        with open(corrected_output_file, 'w', encoding='utf-8') as f:
            f.write(corrected_text)
        print(f"Accent-corrected OCR output saved to: {corrected_output_file}")
        
        # 4. Save words not found in dictionary
        unknown_words_file = base_out_path + "_unknown_words.txt"
        with open(unknown_words_file, 'w', encoding='utf-8') as f:
            sorted_words = sorted(words_not_in_dictionary)
            for word in sorted_words:
                f.write(word + '\n')
        print(f"Words not in dictionary saved to: {unknown_words_file}")
        print(f"Total unknown words: {len(words_not_in_dictionary)}")

    return corrected_text

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <path_to_pdf> [dpi]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    ocr_pdf(pdf_file, dpi=dpi, save_output=True)
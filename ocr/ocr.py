import os
import re
import unicodedata
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import pytesseract
import hunspell
from tqdm import tqdm

# Initialize Hunspell dictionaries for English and Greek
hunspell_en = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
hunspell_el = hunspell.HunSpell('/usr/share/hunspell/el_GR.dic', '/usr/share/hunspell/el_GR.aff')

def remove_greek_accents(text):
    """Remove ALL Greek accents, diacritics, and combining characters"""
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
        (0x1DC0, 0x1DFF),  # Combining diacritical marks supplement
        (0x1AB0, 0x1AFF),  # Combining diacritical marks extended
        (0x1F00, 0x1FFE),  # Greek Extended (includes all polytonic combinations)
    ]
    
    # Create a set of all diacritical marks for faster lookup
    diacritical_chars = {chr(i) for start, end in diacritics_ranges 
                        for i in range(start, end + 1)}
    
    # Direct character mappings for precomposed characters
    accent_map = {
        'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
        'Ά': 'Α', 'Έ': 'Ε', 'Ή': 'Η', 'Ί': 'Ι', 'Ό': 'Ο', 'Ύ': 'Υ', 'Ώ': 'Ω',
        'ΐ': 'ι', 'ΰ': 'υ', 'ϊ': 'ι', 'ϋ': 'υ',
        'ἀ': 'α', 'ἁ': 'α', 'ἂ': 'α', 'ἃ': 'α', 'ἄ': 'α', 'ἅ': 'α', 'ἆ': 'α', 'ἇ': 'α',
        'Ἀ': 'Α', 'Ἁ': 'Α', 'Ἂ': 'Α', 'Ἃ': 'Α', 'Ἄ': 'Α', 'Ἅ': 'Α', 'Ἆ': 'Α', 'Ἇ': 'Α',
        'ἐ': 'ε', 'ἑ': 'ε', 'ἒ': 'ε', 'ἓ': 'ε', 'ἔ': 'ε', 'ἕ': 'ε',
        'Ἐ': 'Ε', 'Ἑ': 'Ε', 'Ἒ': 'Ε', 'Ἓ': 'Ε', 'Ἔ': 'Ε', 'Ἕ': 'Ε',
        'ἠ': 'η', 'ἡ': 'η', 'ἢ': 'η', 'ἣ': 'η', 'ἤ': 'η', 'ἥ': 'η', 'ἦ': 'η', 'ἧ': 'η',
        'Ἠ': 'Η', 'Ἡ': 'Η', 'Ἢ': 'Η', 'Ἣ': 'Η', 'Ἤ': 'Η', 'Ἥ': 'Η', 'Ἦ': 'Η', 'Ἧ': 'Η',
        'ἰ': 'ι', 'ἱ': 'ι', 'ἲ': 'ι', 'ἳ': 'ι', 'ἴ': 'ι', 'ἵ': 'ι', 'ἶ': 'ι', 'ἷ': 'ι',
        'Ἰ': 'Ι', 'Ἱ': 'Ι', 'Ἲ': 'Ι', 'Ἳ': 'Ι', 'Ἴ': 'Ι', 'Ἵ': 'Ι', 'Ἶ': 'Ι', 'Ἷ': 'Ι',
        'ὀ': 'ο', 'ὁ': 'ο', 'ὂ': 'ο', 'ὃ': 'ο', 'ὄ': 'ο', 'ὅ': 'ο',
        'Ὀ': 'Ο', 'Ὁ': 'Ο', 'Ὂ': 'Ο', 'Ὃ': 'Ο', 'Ὄ': 'Ο', 'Ὅ': 'Ο',
        'ὐ': 'υ', 'ὑ': 'υ', 'ὒ': 'υ', 'ὓ': 'υ', 'ὔ': 'υ', 'ὕ': 'υ', 'ὖ': 'υ', 'ὗ': 'υ',
        'Ὑ': 'Υ', 'Ὓ': 'Υ', 'Ὕ': 'Υ', 'Ὗ': 'Υ',
        'ὠ': 'ω', 'ὡ': 'ω', 'ὢ': 'ω', 'ὣ': 'ω', 'ὤ': 'ω', 'ὥ': 'ω', 'ὦ': 'ω', 'ὧ': 'ω',
        'Ὠ': 'Ω', 'Ὡ': 'Ω', 'Ὢ': 'Ω', 'Ὣ': 'Ω', 'Ὤ': 'Ω', 'Ὥ': 'Ω', 'Ὦ': 'Ω', 'Ὧ': 'Ω',
    }
    
    # Process the text in two steps:
    # 1. Replace precomposed characters using the mapping
    result = ''.join(accent_map.get(c, c) for c in text)
    
    # 2. Decompose and remove any remaining combining characters
    decomposed = unicodedata.normalize('NFD', result)
    final = ''.join(c for c in decomposed 
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

def is_greek_text(text):
    """Check if text contains primarily Greek characters"""
    greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
    total_chars = sum(1 for c in text if c.isalpha())
    return total_chars > 0 and greek_chars / total_chars > 0.5

def safe_encode_greek(word):
    """Safely encode Greek word to ISO-8859-7, handling unsupported characters"""
    try:
        return word.encode('iso-8859-7')
    except UnicodeEncodeError:
        # If encoding fails, try to transliterate or remove unsupported characters
        # This is a fallback - you might want to implement more sophisticated handling
        cleaned_word = ''.join(c for c in word if ord(c) < 256)
        try:
            return cleaned_word.encode('iso-8859-7')
        except UnicodeEncodeError:
            return None

def safe_decode_greek(byte_word):
    """Safely decode Greek word from ISO-8859-7 to Unicode"""
    try:
        if isinstance(byte_word, bytes):
            return byte_word.decode('iso-8859-7')
        return byte_word
    except UnicodeDecodeError:
        return None

# Global set to track words not found in dictionary
words_not_in_dictionary = set()

def correct_word_with_accents(word):
    """Spell check word and return properly accented version from dictionary"""
    global words_not_in_dictionary
    norm_word = unicodedata.normalize('NFC', word)
    
    if not norm_word.isalpha():
        return word  # Preserve punctuation, digits, etc.

    # Check if the word is primarily Greek
    if is_greek_text(norm_word):
        try:
            # First try the word as-is
            word_encoded = safe_encode_greek(norm_word)
            if word_encoded is not None and hunspell_el.spell(word_encoded):
                return norm_word
            
            # If not found, try without accents
            stripped_word = remove_greek_accents(norm_word)
            stripped_encoded = safe_encode_greek(stripped_word)
            
            if stripped_encoded is not None:
                if hunspell_el.spell(stripped_encoded):
                    # Word found in dictionary - get suggestions to find accented version
                    suggestions = hunspell_el.suggest(stripped_encoded)
                    if suggestions:
                        # Return first suggestion (should be properly accented)
                        decoded_suggestion = safe_decode_greek(suggestions[0])
                        if decoded_suggestion:
                            return decoded_suggestion
                    # If no suggestions but word is valid, return stripped version
                    return stripped_word
                else:
                    # Not found even without accents
                    words_not_in_dictionary.add(norm_word)
                    return norm_word
            else:
                words_not_in_dictionary.add(norm_word)
                return norm_word
            
        except Exception:
            words_not_in_dictionary.add(norm_word)
            return norm_word
    
    # For English words
    try:
        if hunspell_en.spell(norm_word):
            return norm_word
        
        words_not_in_dictionary.add(norm_word)
        return norm_word
        
    except Exception:
        words_not_in_dictionary.add(norm_word)
        return norm_word

    return norm_word

def correct_text_spelling_preserve_format(text):
    """Apply spell correction while preserving original formatting"""
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
    words = re.findall(word_pattern, text)
    total_words = len(words)
    
    # Create a progress tracker
    progress_counter = {'count': 0}
    
    # Initialize progress bar
    with tqdm(total=total_words, desc="Spell checking", unit="words") as progress_bar:
        corrected_text = re.sub(word_pattern, replace_word_with_progress, text)
    
    return corrected_text

def create_stripped_text(text):
    """Create version with Greek accents removed"""
    return remove_greek_accents(text)

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
        
        # 2. Create and save stripped version (no accents)
        stripped_text = create_stripped_text(full_text)
        stripped_output_file = base_out_path + "_ocr_stripped.txt"
        with open(stripped_output_file, 'w', encoding='utf-8') as f:
            f.write(stripped_text)
        print(f"Stripped OCR output saved to: {stripped_output_file}")

    # 3. Spell-check the stripped version (restoring accents from dictionary)
    corrected_text = correct_text_spelling_preserve_format(full_text)

    if save_output:
        base_out_path = output_path or os.path.splitext(pdf_path)[0]
        corrected_output_file = base_out_path + "_ocr_corrected.txt"
        with open(corrected_output_file, 'w', encoding='utf-8') as f:
            f.write(corrected_text)
        print(f"Spell-corrected OCR output saved to: {corrected_output_file}")
        
        # Save words not found in dictionary
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
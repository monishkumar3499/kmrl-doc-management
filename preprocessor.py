# preprocessor.py

import re
import unicodedata
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocessor")


class TextPreprocessor:
    def __init__(self, lowercase: bool = False, remove_nonprintable: bool = True, 
                 fix_technical_terms: bool = True, preserve_structure: bool = True):
        self.lowercase = lowercase
        self.remove_nonprintable = remove_nonprintable
        self.fix_technical_terms = fix_technical_terms
        self.preserve_structure = preserve_structure
        
        # OCR-specific corrections
        self.ocr_corrections = {
            r'USD\s?O\.': 'USD0.',
            r'INR\s?1\.': 'INR1.',
            r'USD\s?([0-9])': r'USD\1',
            r'INR\s?([0-9])': r'INR\1',
            r'\bGO[I1]': 'GOI',
            r'\bGO[K]': 'GOK',
            r'\bKMR[L1]': 'KMRL',
            r'\bAll[B8]': 'AIIB',
            r'(\d+)\s*percent': r'\1 percent',
            r'(\d+)\s*%': r'\1%',
        }

        # Technical abbreviations (keep spacing intact)
        self.technical_abbrevs = [
            'AFC', 'AIIB', 'KMRL', 'O&M', 'R&R', 'GDP', 'INR', 'USD', 'ECap',
            'EIA', 'CBTC', 'PPE', 'CSR', 'ERP'
        ]

    # ------------------- Basic Cleaning ------------------- #
    def normalize_unicode(self, text: str) -> str:
        """Normalize text for Malayalam + English."""
        return unicodedata.normalize("NFC", text)

    def remove_null_bytes(self, text: str) -> str:
        """Remove null bytes and control characters except newline/tab."""
        text = text.replace("\x00", "")
        text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text

    def remove_non_printable(self, text: str) -> str:
        return ''.join(c for c in text if c.isprintable() or c in "\n\t\r")

    # ------------------- OCR Fixes ------------------- #
    def fix_currency_and_numbers(self, text: str) -> str:
        text = re.sub(r'USD\s*([0-9])', r'USD\1', text)
        text = re.sub(r'INR\s*([0-9])', r'INR\1', text)
        text = re.sub(r'(\d{1,3}),(\d{3})', r'\1,\2', text)
        return text

    def fix_common_ocr_errors(self, text: str) -> str:
        for pattern, repl in self.ocr_corrections.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def fix_technical_abbreviations(self, text: str) -> str:
        for abbrev in self.technical_abbrevs:
            spaced = ' '.join(list(abbrev))
            text = text.replace(spaced, abbrev)
            # Fix I/1 and O/0 misreads
            text = text.replace(abbrev.replace('I', '1'), abbrev)
            text = text.replace(abbrev.replace('O', '0'), abbrev)
        return text

    # ------------------- Malayalam Fixes ------------------- #
    def fix_malayalam_spacing(self, text: str) -> str:
        """Remove unnecessary spaces between Malayalam characters."""
        # Malayalam Unicode range: \u0D00-\u0D7F
        text = re.sub(r'([\u0D00-\u0D7F])\s+([\u0D00-\u0D7F])', r'\1\2', text)
        # Malayalam-English boundaries
        text = re.sub(r'([\u0D00-\u0D7F])\s+([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])\s+([\u0D00-\u0D7F])', r'\1 \2', text)
        return text

    # ------------------- Structure & Formatting ------------------- #
    def preserve_document_structure(self, text: str) -> str:
        if not self.preserve_structure:
            return text
        # Section headers (1. TITLE)
        text = re.sub(r'^(\d+\.)\s*([A-Z][A-Z\s]+)$', r'\1 \2', text, flags=re.MULTILINE)
        # Bullet points
        text = re.sub(r'^\s*[•·▪]\s*', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*-\s*', '- ', text, flags=re.MULTILINE)
        return text

    def fix_table_formatting(self, text: str) -> str:
        text = re.sub(r'\|\s*\|\s*\|', '|||', text)
        text = re.sub(r'\s+\|\s+', ' | ', text)
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        return text

    # ------------------- Cleanup ------------------- #
    def fix_punctuation(self, text: str) -> str:
        text = re.sub(r'``', '"', text)
        text = re.sub(r"''", '"', text)
        text = re.sub(r'--+', '—', text)
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s*([a-zA-Z])', r'\1 \2', text)
        return text

    def remove_extra_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        return text.strip()

    def remove_empty_lines(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                cleaned.append(line.rstrip())
            elif (i > 0 and i < len(lines)-1 and lines[i-1].strip() and lines[i+1].strip()):
                cleaned.append('')
        return '\n'.join(cleaned)

    # ------------------- Main Pipeline ------------------- #
    def preprocess(self, text: str) -> str:
        try:
            logger.info("Starting text preprocessing...")

            text = self.remove_null_bytes(text)
            text = self.normalize_unicode(text)

            if self.remove_nonprintable:
                text = self.remove_non_printable(text)

            if self.fix_technical_terms:
                text = self.fix_technical_abbreviations(text)
                text = self.fix_common_ocr_errors(text)

            text = self.fix_currency_and_numbers(text)
            text = self.fix_malayalam_spacing(text)

            if self.preserve_structure:
                text = self.preserve_document_structure(text)
                text = self.fix_table_formatting(text)

            text = self.fix_punctuation(text)

            if self.lowercase:
                text = text.lower()

            text = self.remove_extra_whitespace(text)
            text = self.remove_empty_lines(text)

            logger.info("Text preprocessing completed successfully")
            return text

        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    # ------------------- Stats ------------------- #
    def get_processing_stats(self, original: str, processed: str) -> Dict[str, any]:
        return {
            'original_length': len(original),
            'processed_length': len(processed),
            'length_change': len(processed) - len(original),
            'original_lines': len(original.splitlines()),
            'processed_lines': len(processed.splitlines()),
            'lines_change': len(processed.splitlines()) - len(original.splitlines())
        }

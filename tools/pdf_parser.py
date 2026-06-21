"""
PDF Resume Parser Tool
Extracts, cleans, and normalizes text content from uploaded PDF files.
"""

import io
import re
from pypdf import PdfReader

# Maximum file size allowed: 5 MB (5 * 1024 * 1024 bytes)
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

class PDFParserError(Exception):
    """Custom exception class for PDF parser errors."""
    pass

def validate_pdf_file(file_bytes: bytes, file_name: str) -> None:
    """
    Validates the uploaded file size and type.
    
    Args:
        file_bytes (bytes): Raw bytes of the uploaded file.
        file_name (str): Original filename.
        
    Raises:
        PDFParserError: If validation fails.
    """
    # 1. Check file size
    file_size = len(file_bytes)
    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        curr_mb = file_size / (1024 * 1024)
        raise PDFParserError(
            f"File size exceeds the limit. Maximum allowed size is {max_mb:.1f}MB. "
            f"Your file is {curr_mb:.2f}MB."
        )

    # 2. Check file extension / signature
    if not file_name.lower().endswith('.pdf'):
        raise PDFParserError("Invalid file format. Only PDF files (.pdf) are supported.")
    
    # Check simple PDF header magic bytes (%PDF-)
    if not file_bytes.startswith(b'%PDF-'):
        raise PDFParserError("The uploaded file does not appear to be a valid PDF document header.")


def clean_extracted_text(text: str) -> str:
    """
    Cleans and normalizes raw text extracted from the PDF.
    
    - Normalizes spacing (collapsing consecutive spaces/tabs).
    - Removes non-printable characters.
    - Normalizes line breaks to single newlines.
    - Strips leading/trailing spaces.
    
    Args:
        text (str): Raw extracted text.
        
    Returns:
        str: Normalized, clean text.
    """
    if not text:
        return ""
    
    # 1. Replace tabs and vertical spaces with standard spaces
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    
    # 2. Normalize multiple newlines to a maximum of two newlines (to preserve paragraphs)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # 3. Strip trailing/leading spaces on each individual line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 4. Strip outer margins of the document
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes, file_name: str) -> str:
    """
    Validates, extracts, and cleans text from a PDF file.
    
    Args:
        file_bytes (bytes): Raw bytes of the PDF.
        file_name (str): Original filename.
        
    Returns:
        str: Extracted and cleaned text from the PDF.
        
    Raises:
        PDFParserError: If the file is invalid, encrypted, or empty.
    """
    # Run validation first
    validate_pdf_file(file_bytes, file_name)

    try:
        # Wrap bytes in a file-like stream
        pdf_stream = io.BytesIO(file_bytes)
        
        # Load PDF using PyPDF
        reader = PdfReader(pdf_stream)
        
        # Check encryption
        if reader.is_encrypted:
            raise PDFParserError("The uploaded PDF is encrypted/password-protected and cannot be parsed.")
        
        # Check page count
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise PDFParserError("The uploaded PDF has no pages.")
        
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)
                
        raw_text = "\n".join(extracted_pages)
        
        # Normalize and clean text
        cleaned_text = clean_extracted_text(raw_text)
        
        # Verify text was successfully extracted (e.g. not a scanned image PDF)
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            raise PDFParserError(
                "Unable to extract text. The PDF might be empty, scanned, or image-only. "
                "Please upload a text-searchable PDF."
            )
            
        return cleaned_text
        
    except PDFParserError as pe:
        # Pass custom errors through
        raise pe
    except Exception as e:
        # Catch unexpected pypdf or stream errors
        raise PDFParserError(f"An unexpected error occurred while parsing the PDF: {str(e)}")

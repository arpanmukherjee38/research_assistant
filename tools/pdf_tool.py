"""
Tool: PDF
Extracts raw text from PDF files using PyMuPDF.
"""

from __future__ import annotations
import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path | str) -> str:
    """
    Extract all text from a local PDF file.
    """
    path_str = str(pdf_path)
    try:
        doc = fitz.open(path_str)
        text = ""
        for page in doc:
            # Extract text block by block for cleaner reading
            text += page.get_text("text") + "\n\n"
        doc.close()
        return text.strip()
    except Exception as exc:
        logger.error("Failed to extract text from %s: %s", path_str, exc)
        return ""


def extract_text_from_bytes(pdf_bytes: bytes, filename: str = "uploaded_doc.pdf") -> str:
    """
    Extract all text from raw PDF bytes (used for Streamlit uploads).
    """
    try:
        # Open the PDF from a memory stream
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n\n"
        doc.close()
        return text.strip()
    except Exception as exc:
        logger.error("Failed to extract text from byte stream (%s): %s", filename, exc)
        return ""
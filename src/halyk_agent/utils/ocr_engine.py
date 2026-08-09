import fitz  # PyMuPDF
import easyocr
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader once to avoid reloading models
# We use Russian and English since the documents are in Kazakhstan and might contain both.
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        logger.info("Initializing EasyOCR reader (ru, en)...")
        # Initialize without forcing GPU if it might cause issues, but easyocr handles fallback.
        _reader = easyocr.Reader(['ru', 'en'], gpu=True) 
    return _reader

def extract_text_from_pdf_ocr(pdf_path: str, max_pages: int = 20, page_numbers: list = None) -> str:
    """
    Extracts text from a PDF by rendering each page to an image and running EasyOCR.
    """
    logger.info(f"Running OCR on {pdf_path}")
    reader = get_reader()
    full_text = []
    
    try:
        doc = fitz.open(pdf_path)
        pages_to_ocr = page_numbers if page_numbers else list(range(min(len(doc), max_pages)))
        
        for i in pages_to_ocr:
            if i >= len(doc): continue
            page = doc.load_page(i)
            # Render page to an image (matrix zoom for better resolution)
            zoom = 2.0  # 2x zoom is usually enough for OCR
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to numpy array for EasyOCR
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # If the image has an alpha channel, drop it
            if pix.n == 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                # grayscale, convert to RGB for easyocr just in case
                img = np.stack((img.squeeze(),)*3, axis=-1)
            
            # Run OCR
            result = reader.readtext(img, detail=0, paragraph=True)
            page_text = "\n".join(result)
            full_text.append(f"--- Page {i+1} ---\n{page_text}")
            
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"OCR failed for {pdf_path}: {e}")
        return ""

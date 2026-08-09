import os
import sys
from pathlib import Path
import concurrent.futures
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from halyk_agent.ingestion.pdf_pipeline import PDFIngestionPipeline, _save_parsed_document

def process_single_pdf(pdf_path: Path, output_dir: Path):
    try:
        # Check if already processed (check for .md file)
        # We need the doc_id to know the exact filename, but doc_id is random UUID for now.
        # Actually, let's just process it anyway or check if ANY file for this pdf's hash exists.
        
        # It's better to process all.
        logger.info(f"Starting {pdf_path.name}")
        pipeline = PDFIngestionPipeline()
        result = pipeline.process_pdf(pdf_path)
        _save_parsed_document(result, output_dir)
        logger.info(f"Finished {pdf_path.name} -> {result.metadata.doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}")
        return False

def main():
    input_dir = Path("E:/AntigravityProjects/halyk-agent/agentic-bank-public/documents")
    output_dir = Path("E:/AntigravityProjects/halyk-agent/data/raw")
    
    if not input_dir.exists():
        logger.error(f"Input dir not found: {input_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process in parallel using 4 workers
    max_workers = 4
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, pdf, output_dir): pdf for pdf in pdf_files}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            logger.info(f"Progress: {completed}/{len(pdf_files)}")

if __name__ == "__main__":
    main()

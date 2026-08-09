"""
PDF Ingestion Pipeline using marker-pdf and docling.
Extracts text, tables with bbox, and metadata.
"""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import fitz  # PyMuPDF for metadata
from loguru import logger

from halyk_agent.config import settings
from halyk_agent.models import (
    BoundingBox,
    DocumentMetadata,
    DocumentType,
    ExtractionMethod,
    ExtractedTable,
    TableCell,
    TextChunk,
)

# Try importing marker-pdf
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    logger.warning("marker-pdf not available, using fallback")

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("docling not available, using fallback")


@dataclass
class ParsedDocument:
    """Container for parsed document results."""
    metadata: DocumentMetadata
    text_chunks: list[TextChunk]
    tables: list[ExtractedTable]
    raw_markdown: str


class PDFIngestionPipeline:
    """Main ingestion pipeline combining marker-pdf and docling."""

    def __init__(self):
        self.chunk_size = settings.ingestion.chunk_size
        self.chunk_overlap = settings.ingestion.chunk_overlap
        self._init_parsers()

    def _init_parsers(self):
        """Initialize parser backends."""
        self.marker_available = MARKER_AVAILABLE
        self.docling_available = DOCLING_AVAILABLE

        if self.docling_available:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            
            # Enable OCR via RapidOCR (PaddleOCR backend)
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = RapidOcrOptions()
            
            self.docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options
                }
            )

    def process_pdf(self, pdf_path: Path) -> ParsedDocument:
        """Process a single PDF file."""
        logger.info(f"Processing PDF: {pdf_path}")

        # Get file hash
        file_hash = self._compute_hash(pdf_path)

        # Get basic metadata via PyMuPDF
        base_meta = self._get_base_metadata(pdf_path, file_hash)

        # Parse with marker (better tables)
        marker_result = None
        if self.marker_available:
            marker_result = self._parse_with_marker(pdf_path)

        # Parse with docling (better structure)
        docling_result = None
        if self.docling_available:
            docling_result = self._parse_with_docling(pdf_path)

        # Merge results
        merged = self._merge_results(base_meta, marker_result, docling_result)

        # Extract metadata using LLM if enabled
        if settings.ingestion.metadata_extraction.use_llm:
            merged.metadata = self._enrich_metadata(merged.metadata, merged.raw_markdown)

        # Create text chunks
        text_chunks = self._create_chunks(merged)

        return ParsedDocument(
            metadata=merged.metadata,
            text_chunks=text_chunks,
            tables=merged.tables,
            raw_markdown=merged.raw_markdown,
        )

    def _get_base_metadata(self, pdf_path: Path, file_hash: str) -> DocumentMetadata:
        """Extract basic metadata using PyMuPDF."""
        doc = fitz.open(pdf_path)
        meta = doc.metadata

        return DocumentMetadata(
            doc_id=str(uuid4()),
            title=meta.get("title") or pdf_path.stem,
            source_path=str(pdf_path),
            page_count=doc.page_count,
            file_hash=file_hash,
            language="ru",  # default, will be refined
        )

    def _parse_with_marker(self, pdf_path: Path) -> dict:
        """Parse with marker-pdf."""
        try:
            converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )
            rendered = converter(str(pdf_path))
            
            try:
                markdown = rendered.markdown
            except AttributeError:
                markdown = text_from_rendered(rendered)
                
            return {
                "markdown": markdown,
                "tables": [],
                "images": getattr(rendered, "images", []),
            }
        except Exception as e:
            logger.error(f"Marker parsing failed: {e}")
            return {"markdown": "", "tables": [], "images": []}

    def _parse_with_docling(self, pdf_path: Path) -> dict:
        """Parse with docling."""
        try:
            result = self.docling_converter.convert(str(pdf_path))
            doc = result.document

            # Export to markdown
            markdown = doc.export_to_markdown()

            # Extract tables
            tables = []
            for table_ix, table in enumerate(doc.tables):
                table_data = self._extract_docling_table(table, table_ix)
                if table_data:
                    tables.append(table_data)

            return {
                "markdown": markdown,
                "tables": tables,
                "document": doc,
            }
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            return {"markdown": "", "tables": [], "document": None}

    def _extract_docling_table(self, table, table_ix: int) -> Optional[ExtractedTable]:
        """Extract table data from docling table object."""
        try:
            # Get bbox
            prov = table.prov[0] if table.prov else None
            if prov:
                bbox = BoundingBox(
                    x0=prov.bbox.l,
                    y0=prov.bbox.b,
                    x1=prov.bbox.r,
                    y1=prov.bbox.t,
                    page=prov.page_no,
                )
            else:
                bbox = BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1)

            # Extract cells
            rows = []
            headers = []
            for row_ix, row in enumerate(table.data.grid):
                cells = []
                for col_ix, cell in enumerate(row):
                    cell_text = cell.text if cell else ""
                    table_cell = TableCell(
                        row=row_ix,
                        col=col_ix,
                        text=cell_text,
                        bbox=bbox,  # approximate
                        is_header=(row_ix == 0 and settings.ingestion.table_extraction.header_detection),
                    )
                    cells.append(table_cell)
                    if row_ix == 0:
                        headers.append(cell_text)
                rows.append(cells)

            return ExtractedTable(
                table_id=str(uuid4()),
                page=bbox.page,
                bbox=bbox,
                headers=headers,
                rows=rows,
                extraction_method=ExtractionMethod.DOCLING_TABLE,
                confidence=0.9,
            )
        except Exception as e:
            logger.error(f"Failed to extract docling table: {e}")
            return None

    def _merge_results(
        self,
        base_meta: DocumentMetadata,
        marker_result: Optional[dict],
        docling_result: Optional[dict]
    ) -> ParsedDocument:
        """Merge results from both parsers, preferring marker for tables."""
        tables = []
        raw_markdown = ""

        # Prefer docling markdown for structure, marker for tables
        if docling_result and docling_result.get("markdown"):
            raw_markdown = docling_result["markdown"]
        elif marker_result and marker_result.get("markdown"):
            raw_markdown = marker_result["markdown"]

        # Merge tables (prefer marker)
        if marker_result and marker_result.get("tables"):
            for table in marker_result["tables"]:
                extracted = self._convert_marker_table(table)
                if extracted:
                    tables.append(extracted)
        elif docling_result and docling_result.get("tables"):
            tables.extend(docling_result["tables"])

        return ParsedDocument(
            metadata=base_meta,
            text_chunks=[],
            tables=tables,
            raw_markdown=raw_markdown,
        )

    def _convert_marker_table(self, marker_table: dict) -> Optional[ExtractedTable]:
        """Convert marker table format to ExtractedTable."""
        try:
            # Marker returns tables as list of lists or similar
            # Adapt based on actual marker output format
            if isinstance(marker_table, list):
                rows = marker_table
            elif isinstance(marker_table, dict) and "cells" in marker_table:
                rows = marker_table["cells"]
            else:
                return None

            if not rows or len(rows) < settings.ingestion.table_extraction.min_rows:
                return None

            # Detect headers
            headers = [str(cell) for cell in rows[0]] if rows else []
            table_rows = []
            for row_ix, row in enumerate(rows):
                cells = []
                for col_ix, cell in enumerate(row):
                    cells.append(TableCell(
                        row=row_ix,
                        col=col_ix,
                        text=str(cell),
                        is_header=(row_ix == 0),
                    ))
                table_rows.append(cells)

            return ExtractedTable(
                table_id=str(uuid4()),
                page=1,  # marker doesn't always give page
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
                headers=headers,
                rows=table_rows,
                extraction_method=ExtractionMethod.MARKER_TABLE,
                confidence=0.85,
            )
        except Exception as e:
            logger.error(f"Failed to convert marker table: {e}")
            return None

    def _create_chunks(self, parsed: ParsedDocument) -> list[TextChunk]:
        """Create text chunks from markdown with overlap."""
        chunks = []
        text = parsed.raw_markdown

        # Simple chunking by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""
        current_page = 1
        chunk_start = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    doc_id=parsed.metadata.doc_id,
                    text=current_chunk,
                    page=current_page,
                    section_header=self._extract_section_header(current_chunk),
                    extraction_method=ExtractionMethod.DOCLING_TEXT if self.docling_available else ExtractionMethod.MARKER_TEXT,
                    metadata=parsed.metadata,
                )
                chunks.append(chunk)

                # Overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # Last chunk
        if current_chunk:
            chunks.append(TextChunk(
                doc_id=parsed.metadata.doc_id,
                text=current_chunk,
                page=current_page,
                section_header=self._extract_section_header(current_chunk),
                extraction_method=ExtractionMethod.DOCLING_TEXT if self.docling_available else ExtractionMethod.MARKER_TEXT,
                metadata=parsed.metadata,
            ))

        return chunks

    def _extract_section_header(self, text: str) -> Optional[str]:
        """Extract section header from text chunk."""
        lines = text.split("\n")
        for line in lines[:3]:
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return None

    def _enrich_metadata(self, metadata: DocumentMetadata, markdown: str) -> DocumentMetadata:
        """Enrich metadata using LLM (placeholder for now)."""
        # TODO: Implement LLM-based metadata extraction
        # This would use an LLM to extract dates, entities, doc_type, etc.
        return metadata

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def process_directory(input_dir: Path, output_dir: Path) -> list[ParsedDocument]:
    """Process all PDFs in a directory."""
    pipeline = PDFIngestionPipeline()
    results = []

    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        try:
            result = pipeline.process_pdf(pdf_path)
            results.append(result)

            # Save intermediate results
            _save_parsed_document(result, output_dir)

        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")

    return results


def _save_parsed_document(parsed: ParsedDocument, output_dir: Path):
    """Save parsed document to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_file = output_dir / f"{parsed.metadata.doc_id}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(parsed.metadata.model_dump(), f, ensure_ascii=False, indent=2, default=str)

    # Save chunks
    chunks_file = output_dir / f"{parsed.metadata.doc_id}_chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in parsed.text_chunks], f, ensure_ascii=False, indent=2, default=str)

    # Save tables
    tables_file = output_dir / f"{parsed.metadata.doc_id}_tables.json"
    with open(tables_file, "w", encoding="utf-8") as f:
        json.dump([t.model_dump() for t in parsed.tables], f, ensure_ascii=False, indent=2, default=str)

    # Save markdown
    md_file = output_dir / f"{parsed.metadata.doc_id}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(parsed.raw_markdown)
---
layout: post
title: Local LLM Document Pipeline Blueprint
description: This guide outlines a blueprint for building a local LLM document pipeline, focusing on efficiency, security, and customization.
date:   2025-03-22 01:42:44 -0500
---
# Building a Local Document Processing Pipeline with LLMs: The Ultimate Architecture

> *"The ability to process, understand, and transform documents is not merely a technical challenge—it is the foundation of knowledge work in the digital age."*

This comprehensive guide presents a production-grade, locally-hosted document processing pipeline that combines elegance with power. By the end, you'll have a system that extracts meaning from documents, structures information intelligently, and enables limitless transformations of your content—all without sending sensitive data to external APIs.

## 📋 Architecture Overview


┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │    │                 │
│  Document       │─→  │  Extraction     │─→  │  Semantic       │─→  │  Storage &      │
│  Ingestion      │    │  Engine         │    │  Processing     │    │  Retrieval      │
│                 │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      ↑                       ↓
                            ┌─────────────────────────┴───────────────────────┐
                            │                                                 │
                            │             Transformation Layer                │
                            │                                                 │
                            └─────────────────────────────────────────────────┘


## 1. High-Fidelity Document Extraction System

The foundation of our pipeline is a robust extraction engine that preserves document structure while efficiently handling multiple formats.

```python
# document_extractor.py
from typing import Dict, Union, List, Optional
import pdfplumber
from docx import Document
import fitz  # PyMuPDF
import logging
import concurrent.futures
from dataclasses import dataclass

@dataclass
class DocumentMetadata:
    """Structured metadata for any document."""
    filename: str
    file_type: str
    page_count: int
    author: Optional[str] = None
    creation_date: Optional[str] = None
    last_modified: Optional[str] = None

@dataclass
class DocumentElement:
    """Represents a structural element of a document."""
    element_type: str  # 'paragraph', 'heading', 'list_item', 'table', etc.
    content: str
    metadata: Dict = None
    position: Dict = None  # For spatial positioning in the document

@dataclass
class DocumentContent:
    """Full representation of a document's content and structure."""
    metadata: DocumentMetadata
    elements: List[DocumentElement]
    raw_text: str = None

class DocumentExtractor:
    """Universal document extraction class with advanced capabilities."""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
    
    def extract(self, file_path: str) -> DocumentContent:
        """Extract content from document with appropriate extractor."""
        lower_path = file_path.lower()
        
        if lower_path.endswith('.pdf'):
            return self._extract_pdf(file_path)
        elif lower_path.endswith('.docx'):
            return self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _extract_pdf(self, file_path: str) -> DocumentContent:
        """Extract content from PDF with advanced structure recognition."""
        try:
            # Using PyMuPDF for metadata and pdfplumber for content
            pdf_doc = fitz.open(file_path)
            metadata = DocumentMetadata(
                filename=file_path.split('/')[-1],
                file_type="pdf",
                page_count=len(pdf_doc),
                author=pdf_doc.metadata.get('author'),
                creation_date=pdf_doc.metadata.get('creationDate'),
                last_modified=pdf_doc.metadata.get('modDate')
            )
            
            elements = []
            raw_text = ""
            
            # Process pages in parallel for large documents
            def process_page(page_num):
                with pdfplumber.open(file_path) as pdf:
                    page = pdf.pages[page_num]
                    page_text = page.extract_text() or ""
                    
                    # Extract tables separately to maintain structure
                    tables = page.extract_tables()
                    
                    # Identify text blocks with their positions
                    blocks = page.extract_words(
                        keep_blank_chars=True,
                        x_tolerance=3,
                        y_tolerance=3,
                        extra_attrs=['fontname', 'size']
                    )
                    
                    page_elements = []
                    
                    # Process text blocks to identify paragraphs and headings
                    current_block = ""
                    current_metadata = {}
                    
                    for word in blocks:
                        # Simplified logic - in production would have more sophisticated
                        # heading/paragraph detection based on font, size, etc.
                        if not current_metadata:
                            current_metadata = {
                                'font': word.get('fontname'),
                                'size': word.get('size'),
                                'page': page_num + 1
                            }
                            
                        if word.get('size') != current_metadata.get('size'):
                            # Font size changed, likely a new element
                            if current_block:
                                element_type = 'heading' if current_metadata.get('size', 0) > 11 else 'paragraph'
                                page_elements.append(DocumentElement(
                                    element_type=element_type,
                                    content=current_block.strip(),
                                    metadata=current_metadata.copy(),
                                    position={'page': page_num + 1}
                                ))
                                current_block = ""
                                current_metadata = {
                                    'font': word.get('fontname'),
                                    'size': word.get('size'),
                                    'page': page_num + 1
                                }
                                
                        current_block += word.get('text', '') + " "
                    
                    # Add the last block
                    if current_block:
                        element_type = 'heading' if current_metadata.get('size', 0) > 11 else 'paragraph'
                        page_elements.append(DocumentElement(
                            element_type=element_type,
                            content=current_block.strip(),
                            metadata=current_metadata,
                            position={'page': page_num + 1}
                        ))
                    
                    # Add tables as structured elements
                    for i, table in enumerate(tables):
                        table_text = "\n".join([" | ".join([cell or "" for cell in row]) for row in table])
                        page_elements.append(DocumentElement(
                            element_type='table',
                            content=table_text,
                            metadata={'table_index': i},
                            position={'page': page_num + 1}
                        ))
                    
                    return page_text, page_elements
            
            # Process pages in parallel for large documents
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_page, i) for i in range(len(pdf_doc))]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            
            # Sort results by page number (they might complete out of order)
            for page_text, page_elements in sorted(results, key=lambda x: x[1][0].position['page'] if x[1] else 0):
                raw_text += page_text + "\n\n"
                elements.extend(page_elements)
            
            return DocumentContent(metadata=metadata, elements=elements, raw_text=raw_text.strip())
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF content: {str(e)}")
            raise
    
    def _extract_docx(self, file_path: str) -> DocumentContent:
        """Extract content from DOCX with structure preservation."""
        try:
            doc = Document(file_path)
            
            # Extract metadata
            metadata = DocumentMetadata(
                filename=file_path.split('/')[-1],
                file_type="docx",
                page_count=0,  # Page count not directly available in python-docx
                author=doc.core_properties.author,
                creation_date=str(doc.core_properties.created) if doc.core_properties.created else None,
                last_modified=str(doc.core_properties.modified) if doc.core_properties.modified else None
            )
            
            elements = []
            raw_text = ""
            
            # Process paragraphs
            for i, para in enumerate(doc.paragraphs):
                if not para.text.strip():
                    continue
                    
                # Determine element type based on paragraph style
                element_type = 'paragraph'
                if para.style.name.startswith('Heading'):
                    element_type = 'heading'
                elif para.style.name.startswith('List'):
                    element_type = 'list_item'
                
                # Extract formatting information
                runs_info = []
                for run in para.runs:
                    runs_info.append({
                        'text': run.text,
                        'bold': run.bold,
                        'italic': run.italic,
                        'underline': run.underline,
                        'font': run.font.name if run.font.name else None
                    })
                
                elements.append(DocumentElement(
                    element_type=element_type,
                    content=para.text,
                    metadata={
                        'style': para.style.name,
                        'runs': runs_info
                    },
                    position={'index': i}
                ))
                
                raw_text += para.text + "\n"
            
            # Process tables
            for i, table in enumerate(doc.tables):
                table_text = ""
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    table_text += row_text + "\n"
                
                elements.append(DocumentElement(
                    element_type='table',
                    content=table_text.strip(),
                    metadata={'table_index': i},
                    position={'index': len(doc.paragraphs) + i}
                ))
                
                raw_text += table_text + "\n\n"
            
            return DocumentContent(metadata=metadata, elements=elements, raw_text=raw_text.strip())
            
        except Exception as e:
            self.logger.error(f"Error extracting DOCX content: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = DocumentExtractor()
    
    # Extract PDF content
    pdf_content = extractor.extract("sample.pdf")
    print(f"PDF Metadata: {pdf_content.metadata}")
    print(f"PDF Elements: {len(pdf_content.elements)}")
    
    # Extract DOCX content
    docx_content = extractor.extract("sample.docx")
    print(f"DOCX Metadata: {docx_content.metadata}")
    print(f"DOCX Elements: {len(docx_content.elements)}")
```

## 2. Semantic Processing with Local LLMs

This module integrates with local LLMs using Ollama while providing a flexible, performant interface that handles model limitations gracefully.

```python
# semantic_processor.py
from typing import Dict, List, Any, Optional, Union
import json
import logging
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from document_extractor import DocumentContent, DocumentElement

class LLMProcessingError(Exception):
    """Raised when there is an error processing content with the LLM."""
    pass

class OllamaClient:
    """Client for interacting with Ollama local LLM server."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "vanilj/Phi-4:latest", 
        timeout: int = 120,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectError))
    )
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from the model with retry logic for robustness."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error: {e}")
            raise LLMProcessingError(f"Failed to get response from LLM: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise LLMProcessingError(f"Error communicating with LLM: {str(e)}")

class SemanticProcessor:
    """Processes document content using a local LLM for intelligent extraction."""
    
    def __init__(
        self, 
        llm_client: OllamaClient = None,
        chunk_size: int = 6000,
        chunk_overlap: int = 1000
    ):
        self.llm_client = llm_client or OllamaClient()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def _chunk_document(self, doc_content: DocumentContent) -> List[str]:
        """Split document into manageable chunks that preserve semantic meaning."""
        elements = doc_content.elements
        chunks = []
        current_chunk = ""
        
        for element in elements:
            # If adding this element would exceed chunk size, save current chunk
            if len(current_chunk) + len(element.content) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep some overlap for context preservation
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text
            
            # Add element content with appropriate formatting
            if element.element_type == 'heading':
                current_chunk += f"\n## {element.content}\n\n"
            elif element.element_type == 'list_item':
                current_chunk += f"• {element.content}\n"
            elif element.element_type == 'table':
                current_chunk += f"\nTABLE:\n{element.content}\n\n"
            else:  # paragraph
                current_chunk += f"{element.content}\n\n"
        
        # Add the final chunk if there's content
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    async def _process_chunk_to_json(self, chunk: str, schema: Dict) -> Dict:
        """Process a document chunk into structured JSON."""
        schema_str = json.dumps(schema, indent=2)
        
        system_prompt = """You are a document structuring expert. 
Your task is to extract information from document text and structure it according to a given schema.
Always respond with valid JSON that exactly matches the provided schema structure."""
        
        user_prompt = f"""Extract structured information from the following document text. 
Format your response as a valid JSON object that strictly follows this schema:

{schema_str}

DOCUMENT TEXT:
{chunk}

Return ONLY the JSON output without any additional text, explanations, or formatting."""
        
        try:
            response = await self.llm_client.generate(user_prompt, system_prompt)
            
            # Find JSON in the response (in case model adds comments)
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")
                    
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # Try to fix common JSON errors
                fixed_response = self._fix_json_response(response)
                return json.loads(fixed_response)
                
        except Exception as e:
            self.logger.error(f"Error processing chunk to JSON: {str(e)}")
            self.logger.error(f"Problematic chunk: {chunk[:100]}...")
            # Return partial data instead of failing completely
            return {"error": str(e), "partial_text": chunk[:100] + "..."}
    
    def _fix_json_response(self, response: str) -> str:
        """Attempt to fix common JSON errors in LLM responses."""
        # Find what looks like the JSON part of the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > 0:
            json_str = response[start_idx:end_idx]
            
            # Common fixes
            # 1. Fix trailing commas before closing braces
            json_str = json_str.replace(',}', '}').replace(',\n}', '\n}')
            json_str = json_str.replace(',]', ']').replace(',\n]', '\n]')
            
            # 2. Fix unescaped quotes in strings
            # This is a simplistic approach - a real implementation would be more sophisticated
            in_string = False
            fixed_chars = []
            
            for i, char in enumerate(json_str):
                if char == '"' and (i == 0 or json_str[i-1] != '\\'):
                    in_string = not in_string
                
                # If we're in a string and find an unescaped quote, escape it
                if in_string and char == '"' and i > 0 and json_str[i-1] != '\\' and i < len(json_str)-1:
                    fixed_chars.append('\\')
                
                fixed_chars.append(char)
            
            return ''.join(fixed_chars)
        
        return response
    
    async def _merge_chunk_results(self, results: List[Dict], schema: Dict) -> Dict:
        """Intelligently merge results from multiple chunks."""
        if not results:
            return {}
            
        # If we only have one chunk, just return it
        if len(results) == 1:
            return results[0]
        
        # For multiple chunks, we need to merge them intelligently
        merged = {}
        
        # Basic strategy - iterate through schema keys and merge accordingly
        for key, value_type in schema.items():
            # String fields: use the non-empty value from the first chunk that has it
            if value_type == "string":
                for result in results:
                    if result.get(key) and isinstance(result.get(key), str) and result[key].strip():
                        merged[key] = result[key]
                        break
                if key not in merged:
                    merged[key] = ""
            
            # List fields: concatenate lists from all chunks and deduplicate
            elif isinstance(value_type, list) or (isinstance(value_type, str) and value_type.startswith("array")):
                all_items = []
                for result in results:
                    if result.get(key) and isinstance(result.get(key), list):
                        all_items.extend(result[key])
                
                # Simple deduplication - this could be more sophisticated
                deduplicated = []
                seen = set()
                for item in all_items:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        deduplicated.append(item)
                
                merged[key] = deduplicated
            
            # Object fields: recursively merge
            elif isinstance(value_type, dict):
                sub_results = [result.get(key, {}) for result in results if isinstance(result.get(key), dict)]
                merged[key] = await self._merge_chunk_results(sub_results, value_type)
            
            # Default case
            else:
                merged[key] = results[0].get(key, "")
        
        return merged
    
    async def process_document(self, doc_content: DocumentContent, schema: Dict) -> Dict:
        """
        Process a document into structured data according to the provided schema.
        
        Args:
            doc_content: The document content object from the extractor
            schema: JSON schema defining the output structure
            
        Returns:
            Dict containing the structured document data
        """
        start_time = time.time()
        self.logger.info(f"Starting document processing: {doc_content.metadata.filename}")
        
        # Split document into manageable chunks
        chunks = self._chunk_document(doc_content)
        self.logger.info(f"Document split into {len(chunks)} chunks")
        
        # Process each chunk in parallel
        chunk_results = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            result = await self._process_chunk_to_json(chunk, schema)
            chunk_results.append(result)
            
        # Merge results from all chunks
        final_result = await self._merge_chunk_results(chunk_results, schema)
        
        # Add document metadata
        final_result["_metadata"] = {
            "filename": doc_content.metadata.filename,
            "file_type": doc_content.metadata.file_type,
            "page_count": doc_content.metadata.page_count,
            "author": doc_content.metadata.author,
            "processing_time": time.time() - start_time
        }
        
        self.logger.info(f"Document processing completed in {time.time() - start_time:.2f} seconds")
        return final_result

# Example schema
DEFAULT_DOCUMENT_SCHEMA = {
    "title": "string",
    "summary": "string",
    "main_topics": ["string"],
    "sections": [
        {
            "heading": "string",
            "content": "string",
            "key_points": ["string"]
        }
    ],
    "entities": {
        "people": ["string"],
        "organizations": ["string"],
        "locations": ["string"],
        "dates": ["string"]
    }
}

# Usage example
async def process_document_example():
    from document_extractor import DocumentExtractor
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    extractor = DocumentExtractor()
    llm_client = OllamaClient(model="vanilj/Phi-4:latest")
    processor = SemanticProcessor(llm_client=llm_client)
    
    # Extract document content
    doc_content = extractor.extract("sample.pdf")
    
    # Process document
    result = await processor.process_document(doc_content, DEFAULT_DOCUMENT_SCHEMA)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(process_document_example())
```

## 3. Robust Storage and Retrieval System

This module provides a flexible data storage layer with support for multiple backends, efficient querying, and versioning.

```python
# document_store.py
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
import sqlite3
import os
import datetime
from dataclasses import dataclass, asdict
from uuid import uuid4
import asyncio
import aiosqlite

@dataclass
class DocumentRecord:
    """Represents a document record in the storage system."""
    doc_id: str
    title: str
    content: Dict[str, Any]  # The structured JSON content
    file_path: str
    file_type: str
    created_at: str
    updated_at: str
    version: int = 1
    tags: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert content to JSON string for storage
        if isinstance(result['content'], dict):
            result['content'] = json.dumps(result['content'])
        if result['tags'] is None:
            result['tags'] = []
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentRecord':
        """Create from dictionary representation."""
        # Parse content from JSON string if needed
        if isinstance(data.get('content'), str):
            try:
                data['content'] = json.loads(data['content'])
            except json.JSONDecodeError:
                # Keep as string if it's not valid JSON
                pass
        
        # Ensure tags is a list
        if data.get('tags') is None:
            data['tags'] = []
            
        return cls(**data)

class DocumentStore:
    """Abstract base class for document storage backends."""
    
    async def initialize(self):
        """Initialize the storage backend."""
        raise NotImplementedError
    
    async def store_document(self, document: DocumentRecord) -> str:
        """Store a document and return its ID."""
        raise NotImplementedError
    
    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document by ID."""
        raise NotImplementedError
    
    async def update_document(self, doc_id: str, content: Dict[str, Any], 
                            increment_version: bool = True) -> Optional[DocumentRecord]:
        """Update a document's content."""
        raise NotImplementedError
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        raise NotImplementedError
    
    async def list_documents(self, limit: int = 100, offset: int = 0, 
                            tags: Optional[List[str]] = None) -> List[DocumentRecord]:
        """List documents with optional filtering."""
        raise NotImplementedError
    
    async def search_documents(self, query: str, 
                             fields: Optional[List[str]] = None) -> List[DocumentRecord]:
        """Search documents by content."""
        raise NotImplementedError
    
    async def get_document_versions(self, doc_id: str) -> List[Dict]:
        """Get all versions of a document."""
        raise NotImplementedError
    
    async def add_tags(self, doc_id: str, tags: List[str]) -> bool:
        """Add tags to a document."""
        raise NotImplementedError
    
    async def close(self):
        """Close the storage connection."""
        raise NotImplementedError

class SQLiteDocumentStore(DocumentStore):
    """SQLite implementation of document storage."""
    
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn = None
    
    async def initialize(self):
        """Initialize the SQLite database."""
        self.logger.info(f"Initializing SQLite document store at {self.db_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        self.conn = await aiosqlite.connect(self.db_path)
        
        # Enable foreign keys
        await self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create documents table
        await self.conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        )
        """)
        
        # Create document versions table
        await self.conn.execute("""
        CREATE TABLE IF NOT EXISTS document_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            content TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
        """)
        
        # Create tags table
        await self.conn.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_name TEXT NOT NULL UNIQUE
        )
        """)
        
        # Create document_tags junction table
        await self.conn.execute("""
        CREATE TABLE IF NOT EXISTS document_tags (
            doc_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (doc_id, tag_id),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        )
        """)
        
        # Create full-text search index
        await self.conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
            doc_id UNINDEXED,
            title,
            content,
            tokenize='porter unicode61'
        )
        """)
        
        # Create triggers to keep FTS index updated
        await self.conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO document_fts(doc_id, title, content)
            VALUES (new.doc_id, new.title, new.content);
        END
        """)
        
        await self.conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            DELETE FROM document_fts WHERE doc_id = old.doc_id;
            INSERT INTO document_fts(doc_id, title, content)
            VALUES (new.doc_id, new.title, new.content);
        END
        """)
        
        await self.conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            DELETE FROM document_fts WHERE doc_id = old.doc_id;
        END
        """)
        
        await self.conn.commit()
        self.logger.info("SQLite document store initialized")
    
    async def store_document(self, document: DocumentRecord) -> str:
        """Store a document and return its ID."""
        if not self.conn:
            await self.initialize()
            
        if not document.doc_id:
            document.doc_id = str(uuid4())
            
        now = datetime.datetime.now().isoformat()
        if not document.created_at:
            document.created_at = now
        if not document.updated_at:
            document.updated_at = now
            
        document_dict = document.to_dict()
        
        try:
            # Insert document
            await self.conn.execute("""
            INSERT INTO documents 
            (doc_id, title, content, file_path, file_type, created_at, updated_at, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_dict['doc_id'],
                document_dict['title'],
                document_dict['content'],
                document_dict['file_path'],
                document_dict['file_type'],
                document_dict['created_at'],
                document_dict['updated_at'],
                document_dict['version']
            ))
            
            # Store initial version
            await self.conn.execute("""
            INSERT INTO document_versions 
            (doc_id, content, version, created_at)
            VALUES (?, ?, ?, ?)
            """, (
                document_dict['doc_id'],
                document_dict['content'],
                document_dict['version'],
                document_dict['created_at']
            ))
            
            # Add tags if present
            if document_dict['tags']:
                await self._add_tags_internal(document_dict['doc_id'], document_dict['tags'])
                
            await self.conn.commit()
            self.logger.info(f"Stored document with ID: {document_dict['doc_id']}")
            return document_dict['doc_id']
            
        except sqlite3.Error as e:
            self.logger.error(f"Error storing document: {str(e)}")
            await self.conn.rollback()
            raise
    
    async def _add_tags_internal(self, doc_id: str, tags: List[str]):
        """Internal method to add tags to a document."""
        for tag in tags:
            # Ensure tag exists in tags table
            cursor = await self.conn.execute(
                "INSERT OR IGNORE INTO tags (tag_name) VALUES (?)", 
                (tag,)
            )
            await self.conn.commit()
            
            # Get tag ID
            cursor = await self.conn.execute(
                "SELECT tag_id FROM tags WHERE tag_name = ?", 
                (tag,)
            )
            row = await cursor.fetchone()
            tag_id = row[0]
            
            # Associate tag with document
            await self.conn.execute(
                "INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?, ?)",
                (doc_id, tag_id)
            )
    
    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document by ID."""
        if not self.conn:
            await self.initialize()
            
        try:
            # Get document
            cursor = await self.conn.execute("""
            SELECT d.doc_id, d.title, d.content, d.file_path, d.file_type, 
                  d.created_at, d.updated_at, d.version
            FROM documents d
            WHERE d.doc_id = ?
            """, (doc_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
                
            # Get tags for document
            cursor = await self.conn.execute("""
            SELECT t.tag_name
            FROM tags t
            JOIN document_tags dt ON t.tag_id = dt.tag_id
            WHERE dt.doc_id = ?
            """, (doc_id,))
            
            tags = [tag[0] for tag in await cursor.fetchall()]
            
            document_dict = {
                'doc_id': row[0],
                'title': row[1],
                'content': row[2],
                'file_path': row[3],
                'file_type': row[4],
                'created_at': row[5],
                'updated_at': row[6],
                'version': row[7],
                'tags': tags
            }
            
            return DocumentRecord.from_dict(document_dict)
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting document: {str(e)}")
            raise
    
    async def update_document(self, doc_id: str, content: Dict[str, Any], 
                            increment_version: bool = True) -> Optional[DocumentRecord]:
        """Update a document's content."""
        if not self.conn:
            await self.initialize()
            
        try:
            # Get current document
            cursor = await self.conn.execute(
                "SELECT version FROM documents WHERE doc_id = ?", 
                (doc_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
                
            current_version = row[0]
            new_version = current_version + 1 if increment_version else current_version
            content_json = json.dumps(content)
            now = datetime.datetime.now().isoformat()
            
            # Update document
            await self.conn.execute("""
            UPDATE documents 
            SET content = ?, updated_at = ?, version = ?
            WHERE doc_id = ?
            """, (content_json, now, new_version, doc_id))
            
            # Store new version if needed
            if increment_version:
                await self.conn.execute("""
                INSERT INTO document_versions 
                (doc_id, content, version, created_at)
                VALUES (?, ?, ?, ?)
                """, (doc_id, content_json, new_version, now))
                
            await self.conn.commit()
            
            # Return updated document
            return await self.get_document(doc_id)
            
        except sqlite3.Error as e:
            self.logger.error(f"Error updating document: {str(e)}")
            await self.conn.rollback()
            raise
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        if not self.conn:
            await self.initialize()
            
        try:
            cursor = await self.conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", 
                (doc_id,)
            )
            await self.conn.commit()
            
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            await self.conn.rollback()
            raise
    
    async def list_documents(self, limit: int = 100, offset: int = 0, 
                            tags: Optional[List[str]] = None) -> List[DocumentRecord]:
        """List documents with optional filtering."""
        if not self.conn:
            await self.initialize()
            
        try:
            documents = []
            
            if tags:
                # Query with tag filtering
                placeholders = ','.join(['?'] * len(tags))
                query = f"""
                SELECT DISTINCT d.doc_id, d.title, d.content, d.file_path, d.file_type, 
                      d.created_at, d.updated_at, d.version
                FROM documents d
                JOIN document_tags dt ON d.doc_id = dt.doc_id
                JOIN tags t ON dt.tag_id = t.tag_id
                WHERE t.tag_name IN ({placeholders})
                ORDER BY d.updated_at DESC
                LIMIT ? OFFSET ?
                """
                cursor = await self.conn.execute(query, (*tags, limit, offset))
            else:
                # Query without tag filtering
                query = """
                SELECT doc_id, title, content, file_path, file_type, 
                      created_at, updated_at, version
                FROM documents
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """
                cursor = await self.conn.execute(query, (limit, offset))
            
            rows = await cursor.fetchall()
            
            for row in rows:
                doc_id = row[0]
                
                # Get tags for document
                cursor = await self.conn.execute("""
                SELECT t.tag_name
                FROM tags t
                JOIN document_tags dt ON t.tag_id = dt.tag_id
                WHERE dt.doc_id = ?
                """, (doc_id,))
                
                doc_tags = [tag[0] for tag in await cursor.fetchall()]
                
                document_dict = {
                    'doc_id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'file_path': row[3],
                    'file_type': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'version': row[7],
                    'tags': doc_tags
                }
                
                documents.append(DocumentRecord.from_dict(document_dict))
            
            return documents
            
        except sqlite3.Error as e:
            self.logger.error(f"Error listing documents: {str(e)}")
            raise
    
    async def search_documents(self, query: str, 
                             fields: Optional[List[str]] = None) -> List[DocumentRecord]:
        """Search documents by content using FTS5."""
        if not self.conn:
            await self.initialize()
            
        try:
            documents = []
            
            # Prepare search parameters
            search_query = ' OR '.join([f"{query}*"] * 3)  # Search with stemming
            
            cursor = await self.conn.execute("""
            SELECT d.doc_id, d.title, d.content, d.file_path, d.file_type, 
                  d.created_at, d.updated_at, d.version
            FROM document_fts fts
            JOIN documents d ON fts.doc_id = d.doc_id
            WHERE document_fts MATCH ?
            ORDER BY rank
            LIMIT 100
            """, (search_query,))
            
            rows = await cursor.fetchall()
            
            for row in rows:
                doc_id = row[0]
                
                # Get tags for document
                cursor = await self.conn.execute("""
                SELECT t.tag_name
                FROM tags t
                JOIN document_tags dt ON t.tag_id = dt.tag_id
                WHERE dt.doc_id = ?
                """, (doc_id,))
                
                doc_tags = [tag[0] for tag in await cursor.fetchall()]
                
                document_dict = {
                    'doc_id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'file_path': row[3],
                    'file_type': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'version': row[7],
                    'tags': doc_tags
                }
                
                documents.append(DocumentRecord.from_dict(document_dict))
            
            return documents
            
        except sqlite3.Error as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            raise
    
    async def get_document_versions(self, doc_id: str) -> List[Dict]:
        """Get all versions of a document."""
        if not self.conn:
            await self.initialize()
            
        try:
            cursor = await self.conn.execute("""
            SELECT content, version, created_at
            FROM document_versions
            WHERE doc_id = ?
            ORDER BY version DESC
            """, (doc_id,))
            
            rows = await cursor.fetchall()
            
            versions = []
            for row in rows:
                version = {
                    'content': row[0],
                    'version': row[1],
                    'created_at': row[2]
                }
                
                # Parse content from JSON string if needed
                if isinstance(version['content'], str):
                    try:
                        version['content'] = json.loads(version['content'])
                    except json.JSONDecodeError:
                        # Keep as string if it's not valid JSON
                        pass
                        
                versions.append(version)
            
            return versions
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting document versions: {str(e)}")
            raise
    
    async def add_tags(self, doc_id: str, tags: List[str]) -> bool:
        """Add tags to a document."""
        if not self.conn:
            await self.initialize()
            
        try:
            # Check if document exists
            cursor = await self.conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", 
                (doc_id,)
            )
            if not await cursor.fetchone():
                return False
                
            await self._add_tags_internal(doc_id, tags)
            await self.conn.commit()
            
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error adding tags: {str(e)}")
            await self.conn.rollback()
            raise
    
    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
            self.logger.info("SQLite document store connection closed")

# Usage example
async def document_store_example():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize store
    store = SQLiteDocumentStore("documents.db")
    await store.initialize()
    
    # Create a document
    doc = DocumentRecord(
        doc_id="",  # Will be auto-generated
        title="Sample Document",
        content={
            "title": "Sample Document",
            "summary": "This is a sample document for testing.",
            "sections": [
                {"heading": "Introduction", "content": "This is the introduction."}
            ]
        },
        file_path="/path/to/sample.pdf",
        file_type="pdf",
        created_at="",  # Will be auto-generated
        updated_at="",  # Will be auto-generated
        tags=["sample", "test"]
    )
    
    # Store document
    doc_id = await store.store_document(doc)
    print(f"Stored document with ID: {doc_id}")
    
    # Retrieve document
    retrieved_doc = await store.get_document(doc_id)
    print(f"Retrieved document: {retrieved_doc.title}")
    
    # Update document
    retrieved_doc.content["summary"] = "Updated summary for testing."
    updated_doc = await store.update_document(doc_id, retrieved_doc.content)
    print(f"Updated document version: {updated_doc.version}")
    
    # List documents
    documents = await store.list_documents(limit=10)
    print(f"Listed {len(documents)} documents")
    
    # Search documents
    search_results = await store.search_documents("sample")
    print(f"Found {len(search_results)} documents matching 'sample'")
    
    # Clean up
    await store.close()

if __name__ == "__main__":
    asyncio.run(document_store_example())
```

## 4. Transformation API with FastAPI

Create a modern, responsive API for document transformations:

```python
# transformation_api.py
from typing import Dict, List, Optional, Any
import logging
import json
import asyncio
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import os

from document_extractor import DocumentExtractor, DocumentContent
from semantic_processor import SemanticProcessor, OllamaClient, DEFAULT_DOCUMENT_SCHEMA
from document_store import SQLiteDocumentStore, DocumentRecord

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for processing, analyzing, and transforming documents using local LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize components
document_extractor = DocumentExtractor()
llm_client = OllamaClient(model="vanilj/Phi-4:latest")
semantic_processor = SemanticProcessor(llm_client=llm_client)
document_store = None  # Will be initialized on startup

# Models
class TransformationRequest(BaseModel):
    doc_id: str
    transformation_type: str = Field(..., description="Type of transformation: 'reword', 'summarize', 'extract_key_points', etc.")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters for the transformation")

class TransformationResponse(BaseModel):
    doc_id: str
    transformation_type: str
    transformed_content: Dict[str, Any]
    execution_time: float

class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    file_type: str
    created_at: str
    updated_at: str
    version: int
    tags: List[str]
    content_preview: str = Field(..., description="Preview of the document content")

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    offset: int = 0

# Dependency for getting the document store
async def get_document_store():
    return document_store

# Background task for processing uploaded documents
async def process_document_task(
    file_path: str,
    file_name: str,
    file_type: str,
    custom_schema: Optional[Dict] = None
):
    try:
        # Extract document content
        logger.info(f"Extracting content from {file_path}")
        doc_content = document_extractor.extract(file_path)
        
        # Process with LLM
        logger.info(f"Processing document with LLM")
        schema = custom_schema or DEFAULT_DOCUMENT_SCHEMA
        result = await semantic_processor.process_document(doc_content, schema)
        
        # Store in database
        logger.info(f"Storing processed document")
        doc = DocumentRecord(
            doc_id="",  # Auto-generated
            title=result.get("title", file_name),
            content=result,
            file_path=file_path,
            file_type=file_type,
            created_at="",  # Auto-generated
            updated_at="",  # Auto-generated
            tags=[]  # No initial tags
        )
        
        doc_id = await document_store.store_document(doc)
        logger.info(f"Document processed and stored with ID: {doc_id}")
        
        # Clean up temporary file if needed
        if os.path.exists(file_path) and "/tmp/" in file_path:
            os.remove(file_path)
            logger.info(f"Temporary file {file_path} removed")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Could implement retry logic or notification system here

# Event handlers
@app.on_event("startup")
async def startup_event():
    global document_store
    logger.info("Initializing document store")
    document_store = SQLiteDocumentStore("documents.db")
    await document_store.initialize()
    logger.info("Document store initialized")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down document store")
    if document_store:
        await document_store.close()
    logger.info("Document store closed")

# Endpoints
@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_schema: Optional[str] = Form(None),
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """Upload and process a document."""
    try:
        # Validate file type
        file_name = file.filename
        if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith('.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Save file temporarily
        file_path = f"/tmp/{int(time.time())}_{file_name}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Parse custom schema if provided
        schema = None
        if custom_schema:
            try:
                schema = json.loads(custom_schema)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON schema")
        
        # Process document in background
        file_type = "pdf" if file_name.lower().endswith('.pdf') else "docx"
        background_tasks.add_task(
            process_document_task, 
            file_path, 
            file_name,
            file_type,
            schema
        )
        
        return {"message": "Document upload successful. Processing started."}
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = 10,
    offset: int = 0,
    tags: Optional[str] = None,
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """List all documents with pagination and optional tag filtering."""
    try:
        tag_list = tags.split(',') if tags else None
        documents = await store.list_documents(limit=limit, offset=offset, tags=tag_list)
        
        # Create response objects with content previews
        response = []
        for doc in documents:
            content_preview = ""
            if isinstance(doc.content, dict):
                # Try to extract a summary or the first section
                if "summary" in doc.content and doc.content["summary"]:
                    content_preview = doc.content["summary"][:200] + "..." if len(doc.content["summary"]) > 200 else doc.content["summary"]
                elif "sections" in doc.content and doc.content["sections"]:
                    first_section = doc.content["sections"][0]
                    if "content" in first_section:
                        content_preview = first_section["content"][:200] + "..." if len(first_section["content"]) > 200 else first_section["content"]
            
            response.append(DocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                file_type=doc.file_type,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                version=doc.version,
                tags=doc.tags or [],
                content_preview=content_preview
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in list_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """Get a document by ID."""
    try:
        document = await store.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/search", response_model=List[DocumentResponse])
async def search_documents(
    search_request: SearchRequest,
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """Search for documents."""
    try:
        documents = await store.search_documents(search_request.query)
        
        # Create response objects with content previews (similar to list_documents)
        response = []
        for doc in documents:
            content_preview = ""
            if isinstance(doc.content, dict):
                if "summary" in doc.content and doc.content["summary"]:
                    content_preview = doc.content["summary"][:200] + "..." if len(doc.content["summary"]) > 200 else doc.content["summary"]
                elif "sections" in doc.content and doc.content["sections"]:
                    first_section = doc.content["sections"][0]
                    if "content" in first_section:
                        content_preview = first_section["content"][:200] + "..." if len(first_section["content"]) > 200 else first_section["content"]
            
            response.append(DocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                file_type=doc.file_type,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                version=doc.version,
                tags=doc.tags or [],
                content_preview=content_preview
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{doc_id}/transform", response_model=TransformationResponse)
async def transform_document(
    doc_id: str,
    request: TransformationRequest,
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """Transform a document with specified transformation type."""
    try:
        start_time = time.time()
        
        # Get document
        document = await store.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Prepare transformation prompt based on type
        transformation_prompts = {
            "reword": "Rewrite the following text to improve clarity and readability while preserving the meaning:",
            "summarize": "Provide a concise summary of the following text:",
            "extract_key_points": "Extract the key points from the following text:",
            "change_tone": f"Rewrite the following text using a {request.parameters.get('tone', 'professional')} tone:",
            "simplify": "Simplify the following text to make it more accessible:"
        }
        
        if request.transformation_type not in transformation_prompts:
            raise HTTPException(status_code=400, detail=f"Unsupported transformation type: {request.transformation_type}")
        
        # Get the content to transform
        content_to_transform = ""
        if request.parameters.get("section_index") is not None:
            # Transform a specific section
            section_index = request.parameters["section_index"]
            if (
                isinstance(document.content, dict) and 
                "sections" in document.content and 
                section_index < len(document.content["sections"])
            ):
                section = document.content["sections"][section_index]
                content_to_transform = section.get("content", "")
            else:
                raise HTTPException(status_code=400, detail="Invalid section index")
        else:
            # Transform the entire document or use the summary
            if isinstance(document.content, dict) and "summary" in document.content:
                content_to_transform = document.content["summary"]
            elif isinstance(document.content, str):
                content_to_transform = document.content
            else:
                # Try to reconstruct from sections
                if isinstance(document.content, dict) and "sections" in document.content:
                    content_to_transform = "\n\n".join([
                        f"## {section.get('heading', 'Section')}\n{section.get('content', '')}"
                        for section in document.content["sections"]
                    ])
        
        if not content_to_transform:
            raise HTTPException(status_code=400, detail="No content available to transform")
        
        # Prepare prompt for the LLM
        prompt = f"{transformation_prompts[request.transformation_type]}\n\n{content_to_transform}"
        
        # Set up system prompt based on transformation type
        system_prompt = "You are an expert at document transformation and improvement."
        
        # Process with LLM
        response = await llm_client.generate(prompt, system_prompt)
        
        # Create transformed content
        transformed_content = {
            "original_length": len(content_to_transform),
            "transformed_length": len(response),
            "transformed_text": response,
            "transformation_type": request.transformation_type
        }
        
        execution_time = time.time() - start_time
        
        # If requested, also update the document with the transformation
        if request.parameters.get("update_document", False):
            # Update the appropriate section
            if request.parameters.get("section_index") is not None:
                section_index = request.parameters["section_index"]
                document.content["sections"][section_index]["content"] = response
            elif "summary" in document.content:
                document.content["summary"] = response
            
            # Save the updated document
            await store.update_document(doc_id, document.content)
        
        return TransformationResponse(
            doc_id=doc_id,
            transformation_type=request.transformation_type,
            transformed_content=transformed_content,
            execution_time=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in transform_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/documents/{doc_id}/tags")
async def add_tags(
    doc_id: str,
    tags: List[str],
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """Add tags to a document."""
    try:
        success = await store.add_tags(doc_id, tags)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Tags added successfully", "doc_id": doc_id, "tags": tags}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_tags: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    store: SQLiteDocumentStore = Depends(get_document_store)
):
    """Delete a document."""
    try:
        success = await store.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully", "doc_id": doc_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("transformation_api:app", host="0.0.0.0", port=8000, reload=True)
```

## 5. Full System Integration with Docker Compose

Bring everything together in a deployable package:

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - LOG_LEVEL=INFO
      - OLLAMA_HOST=ollama
      - OLLAMA_PORT=11434
      - DB_PATH=/app/data/documents.db
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ./ollama-models:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  web:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped
```

## 6. Frontend Interface (React/Next.js)

Create a modern user interface:

```jsx
// App.jsx (simplified version)
import React, { useState, useEffect } from 'react';
import { 
  Container, Box, Typography, TextField, Button, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, Chip, Tab, Tabs, Dialog, DialogContent, DialogTitle, 
  DialogActions, Snackbar, Alert
} from '@mui/material';
import { UploadFile, Search, Transform, Delete } from '@mui/icons-material';

function App() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [transformationType, setTransformationType] = useState('summarize');
  const [transformationResult, setTransformationResult] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/documents');
      const data = await response.json();
      setDocuments(data);
    } catch (error) {
      console.error('Error fetching documents:', error);
      showSnackbar('Failed to load documents', 'error');
    } finally {
      setLoading(false);
    }
  };

  const searchDocuments = async () => {
    if (!searchQuery) {
      fetchDocuments();
      return;
    }
    
    setLoading(true);
    try {
      const response = await fetch('/api/documents/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery })
      });
      const data = await response.json();
      setDocuments(data);
    } catch (error) {
      console.error('Error searching documents:', error);
      showSnackbar('Search failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    setUploadFile(event.target.files[0]);
  };

  const uploadDocument = async () => {
    if (!uploadFile) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', uploadFile);
    
    try {
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        showSnackbar('Document upload started successfully', 'success');
        setUploadFile(null);
        setTimeout(fetchDocuments, 3000); // Refresh after a delay
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Error uploading document:', error);
      showSnackbar(`Upload failed: ${error.message}`, 'error');
    } finally {
      setIsUploading(false);
    }
  };

  const openDocument = async (docId) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/documents/${docId}`);
      const data = await response.json();
      setSelectedDocument(data);
      setDialogOpen(true);
    } catch (error) {
      console.error('Error fetching document:', error);
      showSnackbar('Failed to open document', 'error');
    } finally {
      setLoading(false);
    }
  };

  const transformDocument = async () => {
    if (!selectedDocument) return;
    
    setLoading(true);
    try {
      const response = await fetch(`/api/documents/${selectedDocument.doc_id}/transform`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: selectedDocument.doc_id,
          transformation_type: transformationType,
          parameters: {}
        })
      });
      
      const result = await response.json();
      setTransformationResult(result.transformed_content);
    } catch (error) {
      console.error('Error transforming document:', error);
      showSnackbar('Transformation failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (docId) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
      const response = await fetch(`/api/documents/${docId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        showSnackbar('Document deleted successfully', 'success');
        fetchDocuments();
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Deletion failed');
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      showSnackbar(`Deletion failed: ${error.message}`, 'error');
    }
  };

  const showSnackbar = (message, severity) => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom sx={{ mt: 4 }}>
        Document Processing System
      </Typography>
      
      <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 4 }}>
        <Tab label="All Documents" />
        <Tab label="Upload Document" />
        <Tab label="Search" />
      </Tabs>
      
      {/* Document List Tab */}
      {activeTab === 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Your Documents
          </Typography>
          
          {loading ? (
            <Box display="flex" justifyContent="center" my={4}>
              <CircularProgress />
            </Box>
          ) : (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Title</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Updated</TableCell>
                    <TableCell>Preview</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {documents.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} align="center">
                        No documents found
                      </TableCell>
                    </TableRow>
                  ) : (
                    documents.map(doc => (
                      <TableRow key={doc.doc_id}>
                        <TableCell>{doc.title}</TableCell>
                        <TableCell>
                          <Chip 
                            label={doc.file_type.toUpperCase()} 
                            color={doc.file_type === 'pdf' ? 'error' : 'primary'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{new Date(doc.updated_at).toLocaleDateString()}</TableCell>
                        <TableCell sx={{ maxWidth: 300, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {doc.content_preview}
                        </TableCell>
                        <TableCell>
                          <Button 
                            size="small" 
                            onClick={() => openDocument(doc.doc_id)}
                            sx={{ mr: 1 }}
                          >
                            Open
                          </Button>
                          <Button 
                            size="small" 
                            color="error"
                            onClick={() => deleteDocument(doc.doc_id)}
                          >
                            <Delete fontSize="small" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      )}
      
      {/* Upload Tab */}
      {activeTab === 1 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Upload New Document
          </Typography>
          
          <Box sx={{ border: '1px dashed grey', p: 4, borderRadius: 2, textAlign: 'center', mb: 3 }}>
            <input
              accept=".pdf,.docx"
              style={{ display: 'none' }}
              id="upload-file"
              type="file"
              onChange={handleFileChange}
            />
            <label htmlFor="upload-file">
              <Button 
                variant="outlined" 
                component="span"
                startIcon={<UploadFile />}
              >
                Select File
              </Button>
            </label>
            
            {uploadFile && (
              <Box mt={2}>
                <Typography variant="body1">
                  Selected: {uploadFile.name}
                </Typography>
                <Button 
                  variant="contained" 
                  onClick={uploadDocument}
                  disabled={isUploading}
                  sx={{ mt: 2 }}
                >
                  {isUploading ? <CircularProgress size={24} /> : 'Upload Document'}
                </Button>
              </Box>
            )}
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            Supported formats: PDF, DOCX
          </Typography>
        </Box>
      )}
      
      {/* Search Tab */}
      {activeTab === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Search Documents
          </Typography>
          
          <Box display="flex" mb={3}>
            <TextField
              fullWidth
              label="Search query"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && searchDocuments()}
              variant="outlined"
              sx={{ mr: 2 }}
            />
            <Button 
              variant="contained" 
              onClick={searchDocuments}
              startIcon={<Search />}
            >
              Search
            </Button>
          </Box>
          
          {loading ? (
            <Box display="flex" justifyContent="center" my={4}>
              <CircularProgress />
            </Box>
          ) : (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Title</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Preview</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {documents.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={4} align="center">
                        No results found
                      </TableCell>
                    </TableRow>
                  ) : (
                    documents.map(doc => (
                      <TableRow key={doc.doc_id}>
                        <TableCell>{doc.title}</TableCell>
                        <TableCell>
                          <Chip 
                            label={doc.file_type.toUpperCase()} 
                            color={doc.file_type === 'pdf' ? 'error' : 'primary'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell sx={{ maxWidth: 300, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {doc.content_preview}
                        </TableCell>
                        <TableCell>
                          <Button 
                            size="small" 
                            onClick={() => openDocument(doc.doc_id)}
                          >
                            Open
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      )}
      
      {/* Document Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedDocument && (
          <>
            <DialogTitle>
              {selectedDocument.title}
              {selectedDocument.tags?.map(tag => (
                <Chip 
                  key={tag}
                  label={tag}
                  size="small"
                  sx={{ ml: 1 }}
                />
              ))}
            </DialogTitle>
            <DialogContent dividers>
              <Box mb={3}>
                <Typography variant="subtitle1" gutterBottom>
                  Transform Document
                </Typography>
                <Box display="flex" alignItems="center">
                  <TextField
                    select
                    label="Transformation Type"
                    value={transformationType}
                    onChange={(e) => setTransformationType(e.target.value)}
                    SelectProps={{ native: true }}
                    variant="outlined"
                    sx={{ mr: 2, minWidth: 200 }}
                  >

                    <option value="summarize">Summarize</option>
                    <option value="reword">Reword</option>
                    <option value="extract_key_points">Extract Key Points</option>
                    <option value="change_tone">Change Tone</option>
                    <option value="simplify">Simplify</option>
                  </TextField>
                  <Button 
                    variant="contained" 
                    onClick={transformDocument}
                    startIcon={<Transform />}
                    disabled={loading}
                  >
                    Transform
                  </Button>
                </Box>
              </Box>
              
              {transformationResult && (
                <Box mb={4} p={2} bgcolor="#f5f5f5" borderRadius={1}>
                  <Typography variant="subtitle1" gutterBottom>
                    Transformation Result
                  </Typography>
                  <Typography variant="body1">
                    {transformationResult.transformed_text}
                  </Typography>
                </Box>
              )}
              
              <Typography variant="subtitle1" gutterBottom>
                Document Content
              </Typography>
              
              {selectedDocument.content.summary && (
                <Box mb={3}>
                  <Typography variant="h6">Summary</Typography>
                  <Typography variant="body1">{selectedDocument.content.summary}</Typography>
                </Box>
              )}
              
              {selectedDocument.content.sections?.map((section, index) => (
                <Box key={index} mb={3}>
                  <Typography variant="h6">{section.heading}</Typography>
                  <Typography variant="body1">{section.content}</Typography>
                  
                  {section.key_points?.length > 0 && (
                    <Box mt={2}>
                      <Typography variant="subtitle2">Key Points:</Typography>
                      <ul>
                        {section.key_points.map((point, i) => (
                          <li key={i}>
                            <Typography variant="body2">{point}</Typography>
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Box>
              ))}
              
              {selectedDocument.content.entities && (
                <Box mb={3}>
                  <Typography variant="h6">Entities</Typography>
                  
                  {selectedDocument.content.entities.people?.length > 0 && (
                    <Box mt={1}>
                      <Typography variant="subtitle2">People:</Typography>
                      {selectedDocument.content.entities.people.map((person, i) => (
                        <Chip key={i} label={person} size="small" sx={{ mr: 1, mb: 1 }} />
                      ))}
                    </Box>
                  )}
                  
                  {selectedDocument.content.entities.organizations?.length > 0 && (
                    <Box mt={1}>
                      <Typography variant="subtitle2">Organizations:</Typography>
                      {selectedDocument.content.entities.organizations.map((org, i) => (
                        <Chip key={i} label={org} size="small" sx={{ mr: 1, mb: 1 }} />
                      ))}
                    </Box>
                  )}
                  
                  {selectedDocument.content.entities.locations?.length > 0 && (
                    <Box mt={1}>
                      <Typography variant="subtitle2">Locations:</Typography>
                      {selectedDocument.content.entities.locations.map((loc, i) => (
                        <Chip key={i} label={loc} size="small" sx={{ mr: 1, mb: 1 }} />
                      ))}
                    </Box>
                  )}
                </Box>
              )}
              
              <Box mt={4}>
                <Typography variant="subtitle2" color="text.secondary">
                  File: {selectedDocument.file_path} • 
                  Created: {new Date(selectedDocument.created_at).toLocaleString()} •
                  Version: {selectedDocument.version}
                </Typography>
              </Box>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDialogOpen(false)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default App;
```

## 7. Comprehensive Deployment Guide

Create a `/docs` folder with full documentation:

```markdown
# Local Document Processing Pipeline: Deployment Guide

This guide provides comprehensive instructions for deploying the Local Document Processing Pipeline, a self-contained system for extracting, processing, and transforming documents using local LLMs.

## System Requirements

- **Hardware**:
  - CPU: 4+ cores
  - RAM: 16GB+ (32GB recommended for larger models)
  - Storage: 20GB+ free space
  - GPU: NVIDIA GPU with 8GB+ VRAM (optional but highly recommended)

- **Software**:
  - Docker and Docker Compose
  - NVIDIA Container Toolkit (for GPU acceleration)
  - Git

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-pipeline.git
   cd document-pipeline
   ```

2. Start the system with Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Open your browser and navigate to `http://localhost:3000`

4. The system will automatically download the needed LLM models on first run

## Component Overview

The system consists of three main components:

- **API Server**: Handles document processing, storage, and transformations
- **Ollama**: Runs the local LLM models
- **Web Interface**: Provides a user-friendly interface for the system

## Configuration Options

### Environment Variables

Edit the `.env` file to customize your deployment:

```
# API Server Configuration
LOG_LEVEL=INFO
DB_PATH=/app/data/documents.db
MAX_UPLOAD_SIZE=100MB

# Ollama Configuration
OLLAMA_MODEL=vanilj/Phi-4:latest
OLLAMA_CONCURRENCY=1

# Web Interface Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### LLM Model Selection

By default, the system uses the vanilj/Phi-4 model, which offers a good balance of quality and performance. You can change this by editing the OLLAMA_MODEL variable in the .env file.

Recommended models:
- `vanilj/Phi-4:latest`: Great general-purpose model (4.7GB VRAM)
- `mistral:7b`: Excellent performance for complex text (14GB VRAM)
- `phi3:mini`: Smallest model with decent performance (2.8GB VRAM)

## CPU-Only Deployment

If you don't have a GPU, modify the `docker-compose.yml` file to remove the GPU-specific settings:

```yaml
ollama:
  image: ollama/ollama:latest
  volumes:
    - ./ollama-models:/root/.ollama
  ports:
    - "11434:11434"
  restart: unless-stopped
  # Remove the 'deploy' section for CPU-only mode
```

## Troubleshooting

### Common Issues

1. **System is slow or unresponsive**:
   - Check if your system meets the hardware requirements
   - Try a smaller LLM model
   - Increase Docker container memory limits

2. **Cannot connect to API server**:
   - Check if all containers are running: `docker-compose ps`
   - Check logs: `docker-compose logs api`

3. **Document processing fails**:
   - Check if the Ollama service is running properly
   - Verify that the LLM model was downloaded successfully
   - Check logs: `docker-compose logs ollama`

### Viewing Logs

```bash
# All logs
docker-compose logs

# Specific component logs
docker-compose logs api
docker-compose logs ollama
docker-compose logs web

# Follow logs in real-time
docker-compose logs -f
```

## Scaling for Production

For production environments, consider:

1. **Persistent Storage**: Mount external volumes for database and document storage
2. **Load Balancing**: Deploy multiple API server instances behind a load balancer
3. **Security**: Add proper authentication, HTTPS, and firewall rules
4. **Monitoring**: Implement Prometheus/Grafana for system metrics

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Conclusion: Beyond Document Processing

The architecture presented here goes far beyond a simple document processing system. It represents a paradigm shift in how we interact with documents and knowledge:

1. **Universal Content Extraction**: The system extracts not just raw text but preserves document structure, formatting, and relationships, enabling intelligent processing of any document.

2. **Semantic Understanding**: By integrating local LLMs, the system can comprehend documents at a level approaching human understanding, extracting meaning rather than just data.

3. **Flexible Transformation**: The transformation layer lets users reshape content according to their needs—summarizing dense research papers, simplifying technical documentation, or extracting key insights from lengthy reports.

4. **Self-Contained Intelligence**: By operating entirely locally, this architecture avoids the privacy concerns, costs, and network dependencies of cloud-based solutions.

5. **Extensible Foundation**: This architecture can serve as the foundation for a wide range of knowledge management applications, from research assistants to documentation systems to compliance tools.

This implementation balances elegance with power, providing production-ready code that handles real-world complexity while maintaining clean abstractions. The modular design allows for easy extension and customization, while the Docker-based deployment ensures consistent operation across environments.

By building on this foundation, you can create intelligent document systems that transform how your organization manages and extracts value from information.
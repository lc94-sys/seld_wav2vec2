#!/usr/bin/env python3
"""
Procedure Document Conflict Analyzer
Analyzes multiple .docx procedure documents to find conflicting information.
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import hashlib


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text content from a docx file using pandoc."""
    try:
        result = subprocess.run(
            ['pandoc', docx_path, '-t', 'plain', '--wrap=none'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Warning: Could not extract text from {docx_path}: {result.stderr}")
            return ""
    except Exception as e:
        print(f"Error extracting {docx_path}: {e}")
        return ""


def chunk_text(text: str, max_chars: int = 8000) -> list:
    """Split text into chunks for processing."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_key_procedures(text: str, filename: str) -> dict:
    """Extract key procedural information from document text."""
    return {
        "filename": filename,
        "content": text[:15000],  # Limit content size
        "hash": hashlib.md5(text.encode()).hexdigest()[:8]
    }


def scan_documents(docs_folder: str) -> list:
    """Scan folder for all docx files and extract their content."""
    documents = []
    folder = Path(docs_folder)
    
    docx_files = list(folder.glob("*.docx")) + list(folder.glob("**/*.docx"))
    print(f"Found {len(docx_files)} .docx files")
    
    for i, docx_path in enumerate(docx_files):
        print(f"Processing [{i+1}/{len(docx_files)}]: {docx_path.name}")
        text = extract_text_from_docx(str(docx_path))
        if text:
            doc_info = extract_key_procedures(text, docx_path.name)
            doc_info["path"] = str(docx_path)
            documents.append(doc_info)
    
    return documents


def create_analysis_prompt(doc_batch: list) -> str:
    """Create a prompt for analyzing conflicts between documents."""
    docs_text = ""
    for i, doc in enumerate(doc_batch):
        docs_text += f"\n\n--- DOCUMENT {i+1}: {doc['filename']} ---\n{doc['content'][:8000]}"
    
    return f"""You are analyzing procedure documents to find CONFLICTS - places where two or more documents give contradictory or inconsistent instructions.

IMPORTANT: Only report actual conflicts where documents DISAGREE. Do not report:
- Differences in scope (one document covers topic A, another covers topic B)
- Documents that simply don't mention something
- Minor wording differences that mean the same thing
- Documents covering different procedures entirely

For each REAL conflict found, provide:
1. The specific topic/procedure where conflict exists
2. What Document A says (with document name)
3. What Document B says (with document name)  
4. Why these are contradictory
5. Severity: HIGH (could cause safety/legal issues), MEDIUM (could cause operational issues), LOW (minor inconsistency)

DOCUMENTS TO ANALYZE:
{docs_text}

Respond in this exact JSON format:
{{
    "conflicts_found": true/false,
    "conflicts": [
        {{
            "topic": "Description of the conflicting topic/procedure",
            "document_a": {{
                "name": "filename.docx",
                "states": "What this document says"
            }},
            "document_b": {{
                "name": "filename.docx", 
                "states": "What this document says"
            }},
            "explanation": "Why these conflict",
            "severity": "HIGH/MEDIUM/LOW"
        }}
    ],
    "summary": "Brief overall summary"
}}

If no conflicts found, return:
{{
    "conflicts_found": false,
    "conflicts": [],
    "summary": "No conflicts detected between these documents"
}}"""


def save_documents_json(documents: list, output_path: str):
    """Save extracted documents to JSON for later analysis."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(documents)} documents to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract text from procedure documents')
    parser.add_argument('docs_folder', help='Folder containing .docx files')
    parser.add_argument('--output', '-o', default='extracted_docs.json', help='Output JSON file')
    args = parser.parse_args()
    
    if not os.path.isdir(args.docs_folder):
        print(f"Error: {args.docs_folder} is not a valid directory")
        return
    
    documents = scan_documents(args.docs_folder)
    
    if documents:
        save_documents_json(documents, args.output)
        print(f"\nExtracted {len(documents)} documents")
        print(f"Next step: Run conflict analysis on {args.output}")
    else:
        print("No documents found or extracted")


if __name__ == "__main__":
    main()

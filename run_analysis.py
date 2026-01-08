#!/usr/bin/env python3
"""
Procedure Document Conflict Analyzer - Main Runner
Orchestrates the full workflow: extract → analyze → report

Supports multiple LLM providers: OpenAI, Anthropic, Ollama, Azure
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import our modules
from procedure_conflict_analyzer import scan_documents, save_documents_json
from analyze_conflicts import run_analysis, LLM_PROVIDER
from generate_report import generate_html_report
from generate_excel_report import create_excel_report


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     PROCEDURE DOCUMENT CONFLICT ANALYZER                     ║
║     Finds conflicting information across your documents      ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_full_analysis(docs_folder: str, output_dir: str = ".", provider: str = None):
    """Run the complete analysis pipeline."""
    
    print_banner()
    
    # Set provider if specified
    if provider:
        import analyze_conflicts
        analyze_conflicts.LLM_PROVIDER = provider
        print(f"Using LLM provider: {provider}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Extract text from documents
    print("\n📄 STEP 1: Extracting text from documents...")
    print("-" * 50)
    
    documents = scan_documents(docs_folder)
    
    if not documents:
        print("❌ No documents found or extracted. Please check your folder path.")
        return None
    
    extracted_json = output_path / f"extracted_docs_{timestamp}.json"
    save_documents_json(documents, str(extracted_json))
    print(f"✅ Extracted {len(documents)} documents")
    
    # Step 2: Analyze for conflicts
    print("\n🔍 STEP 2: Analyzing documents for conflicts...")
    print("-" * 50)
    
    results = run_analysis(str(extracted_json))
    
    conflict_json = output_path / f"conflict_analysis_{timestamp}.json"
    with open(conflict_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Analysis complete: {results['total_conflicts']} conflicts found")
    
    # Step 3: Generate reports
    print("\n📊 STEP 3: Generating reports...")
    print("-" * 50)
    
    # HTML Report
    html_report = output_path / f"conflict_report_{timestamp}.html"
    html_content = generate_html_report(results)
    with open(html_report, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ HTML report: {html_report}")
    
    # Excel Report
    excel_report = output_path / f"conflict_report_{timestamp}.xlsx"
    create_excel_report(results, str(excel_report))
    print(f"✅ Excel report: {excel_report}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Documents Analyzed: {results['total_documents']}")
    print(f"Total Conflicts:    {results['total_conflicts']}")
    print(f"  • HIGH severity:   {results['conflicts_by_severity']['HIGH']}")
    print(f"  • MEDIUM severity: {results['conflicts_by_severity']['MEDIUM']}")
    print(f"  • LOW severity:    {results['conflicts_by_severity']['LOW']}")
    print("\n📁 Output Files:")
    print(f"  • {conflict_json}")
    print(f"  • {html_report}")
    print(f"  • {excel_report}")
    print("=" * 60)
    
    return {
        "results": results,
        "files": {
            "json": str(conflict_json),
            "html": str(html_report),
            "excel": str(excel_report)
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze procedure documents for conflicting information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LLM Providers:
  openai     - Uses OPENAI_API_KEY env var (default model: gpt-4o)
  anthropic  - Uses ANTHROPIC_API_KEY env var
  ollama     - Local Ollama instance (default: llama3.1:70b)
  azure      - Azure OpenAI (requires AZURE_ENDPOINT env var)

Examples:
  # Using OpenAI (default)
  export OPENAI_API_KEY="sk-..."
  python run_analysis.py /path/to/docs
  
  # Using local Ollama
  python run_analysis.py /path/to/docs --provider ollama
  
  # Using Anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
  python run_analysis.py /path/to/docs --provider anthropic
        """
    )
    
    parser.add_argument('docs_folder', help='Folder containing .docx procedure documents')
    parser.add_argument('--output', '-o', default='.', help='Output directory for reports')
    parser.add_argument('--provider', '-p', 
                        choices=['openai', 'anthropic', 'ollama', 'azure'],
                        default='openai',
                        help='LLM provider to use (default: openai)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.docs_folder):
        print(f"❌ Error: '{args.docs_folder}' is not a valid directory")
        sys.exit(1)
    
    run_full_analysis(args.docs_folder, args.output, args.provider)


if __name__ == "__main__":
    main()

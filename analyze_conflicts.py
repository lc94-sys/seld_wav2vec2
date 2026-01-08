#!/usr/bin/env python3
"""
Conflict Detection Script
LLM-agnostic - supports multiple providers (OpenAI, Anthropic, Ollama, Azure, etc.)
"""

import json
import os
import sys
from itertools import combinations
from datetime import datetime


# ============================================================
# LLM PROVIDER CONFIGURATION - EDIT THIS SECTION
# ============================================================

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")  # openai, anthropic, ollama, azure

# Provider-specific settings
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")

# ============================================================


def load_documents(json_path: str) -> list:
    """Load extracted documents from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_batches(documents: list, batch_size: int = 3) -> list:
    """Create batches of documents for comparison. Smaller batches for 8B models."""
    batches = []
    n = len(documents)
    
    if n <= batch_size:
        return [documents]
    
    for i in range(0, n, batch_size - 1):
        batch = documents[i:i + batch_size]
        if len(batch) >= 2:
            batches.append(batch)
    
    return batches


def get_conflict_prompt(batch: list) -> str:
    """Generate the analysis prompt for a batch of documents."""
    docs_text = ""
    for i, doc in enumerate(batch):
        content = doc['content'][:4000]  # Reduced for 8B model
        docs_text += f"\n\n=== DOCUMENT {i+1}: {doc['filename']} ===\n{content}"
    
    return f"""Analyze these procedure documents for CONFLICTS - places where documents give contradictory instructions.

ONLY report actual conflicts where documents DISAGREE on the same topic. Do NOT report:
- Topics only covered by one document
- Minor wording differences with same meaning
- Different scopes or procedures

For each conflict, identify:
1. Topic where conflict exists
2. What each document states
3. Why they contradict
4. Severity: HIGH (safety/legal), MEDIUM (operational), LOW (minor)

DOCUMENTS:
{docs_text}

Respond in JSON format ONLY (no other text):
{{
    "conflicts_found": true/false,
    "conflicts": [
        {{
            "topic": "Conflicting topic",
            "document_a": {{"name": "file.docx", "states": "What it says"}},
            "document_b": {{"name": "file.docx", "states": "What it says"}},
            "explanation": "Why they conflict",
            "severity": "HIGH/MEDIUM/LOW"
        }}
    ]
}}"""


def call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str) -> str:
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_ollama(prompt: str) -> str:
    """Call local Ollama instance."""
    import requests
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=300
    )
    return response.json()["response"]


def call_azure(prompt: str) -> str:
    """Call Azure OpenAI."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_version="2024-02-15-preview"
    )
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000
    )
    return response.choices[0].message.content


def call_llm(prompt: str) -> str:
    """Route to the appropriate LLM provider."""
    providers = {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "ollama": call_ollama,
        "azure": call_azure
    }
    
    if LLM_PROVIDER not in providers:
        raise ValueError(f"Unknown provider: {LLM_PROVIDER}. Use: {list(providers.keys())}")
    
    return providers[LLM_PROVIDER](prompt)


def parse_llm_response(response_text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        # Try direct parse first
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0]
    else:
        json_str = response_text
    
    return json.loads(json_str.strip())


def analyze_batch(batch: list) -> dict:
    """Analyze a batch of documents for conflicts."""
    prompt = get_conflict_prompt(batch)
    
    try:
        response_text = call_llm(prompt)
        return parse_llm_response(response_text)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse response as JSON: {e}")
        return {"conflicts_found": False, "conflicts": [], "error": str(e)}
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"conflicts_found": False, "conflicts": [], "error": str(e)}


def deduplicate_conflicts(all_conflicts: list) -> list:
    """Remove duplicate conflicts found across batches."""
    unique = []
    seen_keys = set()
    
    for conflict in all_conflicts:
        docs = sorted([
            conflict.get('document_a', {}).get('name', ''),
            conflict.get('document_b', {}).get('name', '')
        ])
        key = f"{docs[0]}|{docs[1]}|{conflict.get('topic', '')[:50]}"
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(conflict)
    
    return unique


def run_analysis(json_path: str) -> dict:
    """Run the full conflict analysis."""
    
    print(f"Using LLM provider: {LLM_PROVIDER}")
    
    # Load documents
    print(f"Loading documents from {json_path}...")
    documents = load_documents(json_path)
    print(f"Loaded {len(documents)} documents")
    
    # Create batches
    batches = create_batches(documents, batch_size=5)
    print(f"Created {len(batches)} batches for analysis")
    
    # Analyze each batch
    all_conflicts = []
    for i, batch in enumerate(batches):
        print(f"Analyzing batch {i+1}/{len(batches)}...")
        result = analyze_batch(batch)
        if result.get('conflicts_found') and result.get('conflicts'):
            all_conflicts.extend(result['conflicts'])
    
    # Deduplicate
    unique_conflicts = deduplicate_conflicts(all_conflicts)
    
    # Sort by severity
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    unique_conflicts.sort(key=lambda x: severity_order.get(x.get('severity', 'LOW'), 3))
    
    return {
        "analysis_date": datetime.now().isoformat(),
        "total_documents": len(documents),
        "total_conflicts": len(unique_conflicts),
        "conflicts_by_severity": {
            "HIGH": len([c for c in unique_conflicts if c.get('severity') == 'HIGH']),
            "MEDIUM": len([c for c in unique_conflicts if c.get('severity') == 'MEDIUM']),
            "LOW": len([c for c in unique_conflicts if c.get('severity') == 'LOW'])
        },
        "conflicts": unique_conflicts
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze documents for conflicts')
    parser.add_argument('json_file', help='JSON file with extracted documents')
    parser.add_argument('--output', '-o', default='conflict_report.json', help='Output JSON file')
    parser.add_argument('--provider', '-p', help='LLM provider (openai, anthropic, ollama, azure)')
    args = parser.parse_args()
    
    # Override provider if specified
    if args.provider:
        global LLM_PROVIDER
        LLM_PROVIDER = args.provider
    
    results = run_analysis(args.json_file)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Documents analyzed: {results['total_documents']}")
    print(f"Conflicts found: {results['total_conflicts']}")
    print(f"  - HIGH severity: {results['conflicts_by_severity']['HIGH']}")
    print(f"  - MEDIUM severity: {results['conflicts_by_severity']['MEDIUM']}")
    print(f"  - LOW severity: {results['conflicts_by_severity']['LOW']}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

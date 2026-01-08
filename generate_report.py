#!/usr/bin/env python3
"""
Generate formatted HTML report from conflict analysis results.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Procedure Document Conflict Report</title>
    <style>
        :root {
            --high: #dc3545;
            --medium: #fd7e14;
            --low: #ffc107;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #212529;
            --border: #dee2e6;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        
        .container { max-width: 1200px; margin: 0 auto; }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        
        h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        
        .meta { opacity: 0.9; font-size: 0.9rem; }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-card h3 { color: #6c757d; font-size: 0.85rem; text-transform: uppercase; }
        .stat-card .value { font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; }
        .stat-card.high .value { color: var(--high); }
        .stat-card.medium .value { color: var(--medium); }
        .stat-card.low .value { color: var(--low); }
        
        .filters {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .filters label { font-weight: 500; }
        
        .filters select, .filters input {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 1rem;
        }
        
        .conflict-card {
            background: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }
        
        .conflict-header {
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        
        .conflict-header h3 { font-size: 1.1rem; }
        
        .severity-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
            color: white;
        }
        
        .severity-badge.high { background: var(--high); }
        .severity-badge.medium { background: var(--medium); }
        .severity-badge.low { background: var(--low); color: #212529; }
        
        .conflict-body { padding: 1.5rem; }
        
        .documents-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1rem;
        }
        
        @media (max-width: 768px) {
            .documents-comparison { grid-template-columns: 1fr; }
        }
        
        .doc-box {
            background: var(--bg);
            padding: 1rem;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        
        .doc-box h4 {
            font-size: 0.85rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
        }
        
        .doc-name {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .doc-content {
            font-size: 0.95rem;
            color: var(--text);
        }
        
        .explanation {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
        }
        
        .explanation h4 {
            font-size: 0.85rem;
            color: #856404;
            margin-bottom: 0.5rem;
        }
        
        .no-conflicts {
            text-align: center;
            padding: 4rem 2rem;
            background: var(--card-bg);
            border-radius: 8px;
        }
        
        .no-conflicts h2 { color: #28a745; margin-bottom: 1rem; }
        
        .highlight {
            background: #fff3cd;
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 Procedure Document Conflict Report</h1>
            <div class="meta">
                Generated: {generation_date}<br>
                Documents Analyzed: {total_documents}
            </div>
        </header>
        
        <div class="summary">
            <div class="stat-card">
                <h3>Total Conflicts</h3>
                <div class="value">{total_conflicts}</div>
            </div>
            <div class="stat-card high">
                <h3>High Severity</h3>
                <div class="value">{high_count}</div>
            </div>
            <div class="stat-card medium">
                <h3>Medium Severity</h3>
                <div class="value">{medium_count}</div>
            </div>
            <div class="stat-card low">
                <h3>Low Severity</h3>
                <div class="value">{low_count}</div>
            </div>
        </div>
        
        <div class="filters">
            <label>Filter by Severity:</label>
            <select id="severityFilter" onchange="filterConflicts()">
                <option value="all">All</option>
                <option value="HIGH">High Only</option>
                <option value="MEDIUM">Medium Only</option>
                <option value="LOW">Low Only</option>
            </select>
            <label>Search:</label>
            <input type="text" id="searchBox" placeholder="Search conflicts..." oninput="filterConflicts()">
        </div>
        
        <div id="conflicts-container">
            {conflicts_html}
        </div>
        
        <footer>
            Procedure Document Conflict Analyzer | Report generated using Claude AI
        </footer>
    </div>
    
    <script>
        function filterConflicts() {
            const severity = document.getElementById('severityFilter').value;
            const search = document.getElementById('searchBox').value.toLowerCase();
            const cards = document.querySelectorAll('.conflict-card');
            
            cards.forEach(card => {
                const cardSeverity = card.dataset.severity;
                const cardText = card.textContent.toLowerCase();
                
                const matchesSeverity = severity === 'all' || cardSeverity === severity;
                const matchesSearch = search === '' || cardText.includes(search);
                
                card.style.display = (matchesSeverity && matchesSearch) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>"""


CONFLICT_CARD_TEMPLATE = """
<div class="conflict-card" data-severity="{severity}">
    <div class="conflict-header">
        <h3>{topic}</h3>
        <span class="severity-badge {severity_lower}">{severity}</span>
    </div>
    <div class="conflict-body">
        <div class="documents-comparison">
            <div class="doc-box">
                <h4>Document A</h4>
                <div class="doc-name">{doc_a_name}</div>
                <div class="doc-content">{doc_a_states}</div>
            </div>
            <div class="doc-box">
                <h4>Document B</h4>
                <div class="doc-name">{doc_b_name}</div>
                <div class="doc-content">{doc_b_states}</div>
            </div>
        </div>
        <div class="explanation">
            <h4>⚠️ Why This Conflicts</h4>
            {explanation}
        </div>
    </div>
</div>
"""


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def generate_html_report(data: dict) -> str:
    """Generate HTML report from conflict data."""
    
    conflicts_html = ""
    
    if data['total_conflicts'] == 0:
        conflicts_html = """
        <div class="no-conflicts">
            <h2>✅ No Conflicts Found</h2>
            <p>All analyzed documents appear to be consistent with each other.</p>
        </div>
        """
    else:
        for conflict in data['conflicts']:
            doc_a = conflict.get('document_a', {})
            doc_b = conflict.get('document_b', {})
            
            conflicts_html += CONFLICT_CARD_TEMPLATE.format(
                topic=escape_html(conflict.get('topic', 'Unknown Topic')),
                severity=conflict.get('severity', 'LOW'),
                severity_lower=conflict.get('severity', 'LOW').lower(),
                doc_a_name=escape_html(doc_a.get('name', 'Unknown')),
                doc_a_states=escape_html(doc_a.get('states', 'Not specified')),
                doc_b_name=escape_html(doc_b.get('name', 'Unknown')),
                doc_b_states=escape_html(doc_b.get('states', 'Not specified')),
                explanation=escape_html(conflict.get('explanation', 'No explanation provided'))
            )
    
    severity = data.get('conflicts_by_severity', {})
    
    return HTML_TEMPLATE.format(
        generation_date=data.get('analysis_date', datetime.now().isoformat()),
        total_documents=data.get('total_documents', 0),
        total_conflicts=data.get('total_conflicts', 0),
        high_count=severity.get('HIGH', 0),
        medium_count=severity.get('MEDIUM', 0),
        low_count=severity.get('LOW', 0),
        conflicts_html=conflicts_html
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <conflict_report.json> [output.html]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "conflict_report.html"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    html = generate_html_report(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()

# Procedure Document Conflict Analyzer

Analyzes multiple .docx procedure documents to identify conflicting information, then generates well-formatted reports highlighting the conflicts.

**Supports multiple LLM providers:** OpenAI, Anthropic, Ollama (local), Azure OpenAI

## Quick Start

```bash
# Using OpenAI (default)
export OPENAI_API_KEY="sk-..."
python run_analysis.py /path/to/your/documents/folder

# Using local Ollama
python run_analysis.py /path/to/docs --provider ollama

# Using Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python run_analysis.py /path/to/docs --provider anthropic

# Specify output location
python run_analysis.py /path/to/docs --output ./reports --provider openai
```

## Supported LLM Providers

| Provider | Env Variable | Default Model | Notes |
|----------|--------------|---------------|-------|
| `openai` | `OPENAI_API_KEY` | gpt-4o | Default provider |
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet | |
| `ollama` | (none needed) | llama3.1:70b | Local, set `OLLAMA_BASE_URL` if not localhost |
| `azure` | `AZURE_OPENAI_API_KEY` | gpt-4o | Also set `AZURE_ENDPOINT` |

### Customizing Models

Set environment variables to use different models:
```bash
export OPENAI_MODEL="gpt-4-turbo"
export OLLAMA_MODEL="mixtral:8x7b"
export OLLAMA_BASE_URL="http://192.168.1.100:11434"
```

## What It Does

1. **Extracts** text from all `.docx` files in your folder (and subfolders)
2. **Analyzes** documents in batches using your chosen LLM to find conflicts
3. **Generates** formatted reports:
   - **HTML Report** - Interactive, filterable web page
   - **Excel Report** - Spreadsheet with summary and detail sheets

## Output Reports

### HTML Report
- Visual, color-coded conflict cards
- Filter by severity (HIGH/MEDIUM/LOW)
- Search functionality
- Side-by-side document comparison

### Excel Report
Three sheets:
- **Summary** - Overview statistics
- **Conflicts Detail** - All conflicts with full details
- **HIGH Priority** - Quick reference for critical conflicts

## Conflict Severity Levels

| Level | Description |
|-------|-------------|
| **HIGH** | Could cause safety, legal, or compliance issues |
| **MEDIUM** | Could cause operational problems or confusion |
| **LOW** | Minor inconsistencies in wording or approach |

## Requirements

- Python 3.8+
- pandoc (for text extraction)
- LLM provider credentials (see above)

### Install Dependencies

```bash
# Core dependencies
pip install openpyxl requests

# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For Ollama - no extra packages needed (uses requests)

# For Azure OpenAI
pip install openai
```

## Individual Scripts

You can also run each step separately:

```bash
# Step 1: Extract text from documents
python procedure_conflict_analyzer.py /path/to/docs --output extracted.json

# Step 2: Analyze for conflicts
python analyze_conflicts.py extracted.json --output conflicts.json

# Step 3: Generate HTML report
python generate_report.py conflicts.json conflict_report.html

# Step 4: Generate Excel report  
python generate_excel_report.py conflicts.json conflict_report.xlsx
```

## How It Works

### Document Processing
- Documents are processed in batches of 5 for thorough cross-comparison
- Text is chunked to fit within API limits
- Overlapping batches ensure all document pairs are compared

### Conflict Detection
The AI looks for actual contradictions where documents:
- Give different instructions for the same procedure
- Specify conflicting requirements, timeframes, or values
- Have incompatible steps or processes

It does **NOT** flag:
- Documents covering different topics
- Minor wording differences with same meaning
- Missing information (one doc covers topic, another doesn't)

### Deduplication
Conflicts found across multiple batches are automatically deduplicated.

## Tips for Best Results

1. **Organize documents** - Put related procedures in the same folder
2. **Name files clearly** - Descriptive filenames help identify conflicts
3. **200 docs is fine** - The tool batches efficiently for large document sets
4. **Review HIGH first** - Start with high-severity conflicts

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║     PROCEDURE DOCUMENT CONFLICT ANALYZER                     ║
╚══════════════════════════════════════════════════════════════╝

📄 STEP 1: Extracting text from documents...
Found 47 .docx files
Processing [1/47]: safety_procedure_v1.docx
...
✅ Extracted 47 documents

🔍 STEP 2: Analyzing documents for conflicts...
Analyzing batch 1/10...
...
✅ Analysis complete: 12 conflicts found

📊 STEP 3: Generating reports...
✅ HTML report: conflict_report_20240115_143022.html
✅ Excel report: conflict_report_20240115_143022.xlsx

═══════════════════════════════════════════════════════════════
📋 ANALYSIS SUMMARY
═══════════════════════════════════════════════════════════════
Documents Analyzed: 47
Total Conflicts:    12
  • HIGH severity:   2
  • MEDIUM severity: 6
  • LOW severity:    4
═══════════════════════════════════════════════════════════════
```

## Troubleshooting

**"No documents found"**
- Check the folder path is correct
- Ensure files have `.docx` extension (not `.doc`)

**"API Error"**
- Verify your ANTHROPIC_API_KEY is set correctly
- Check your API quota

**"Could not extract text"**
- Ensure pandoc is installed: `apt install pandoc` or `brew install pandoc`
- Check the file isn't corrupted

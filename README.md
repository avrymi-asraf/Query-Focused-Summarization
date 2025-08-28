# Query-Focused-Summarization




# Project Structure

- `src/` - Main source code (agents, workflow)
- `articals/` - Example articles
- `requirements.txt` - Python dependencies

# Usage

1. Install requirements: `pip install -r requirements.txt`
2. Run: `python src/main.py --file <path_to_article> --query "<your_query>"`

## Command Line Arguments

- `--file`: Path to the article file (required)
- `--query`: Query for summarization (required)
- `--max_iterations`: Maximum number of iterations (default: 5)
- `--output_format`: Output format - `print` for console output or `json` for structured data (default: print)

## Examples

### Console Output (default)
```bash
python src/main.py --file articals/hebrew-university.md --query "What are the main research areas?" --max_iterations 3
```

### JSON Output
```bash
python src/main.py --file articals/hebrew-university.md --query "What are the main research areas?" --max_iterations 3 --output_format json
```
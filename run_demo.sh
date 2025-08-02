#!/bin/bash
# Simple script to run the Query-Focused Summarization workflow

ARTICLE_PATH="articals/hebrew-university.md"
QUERY="What are the main research areas at Hebrew University?"
MAX_ITER=3
OUTPUT_FORMAT="print"  # Options: "print" or "json"

echo "Running Query-Focused Summarization Demo..."
echo "Article: $ARTICLE_PATH"
echo "Query: $QUERY"
echo "Max Iterations: $MAX_ITER"
echo "Output Format: $OUTPUT_FORMAT"
echo "---"

source .venv/bin/activate && python3 src/main.py --file "$ARTICLE_PATH" --query "$QUERY" --max_iterations "$MAX_ITER" --output_format "$OUTPUT_FORMAT"

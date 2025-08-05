#!/bin/bash
# Simple script to run the Query-Focused Summarization workflow

ARTICLE_PATH="articals/The Linear Representation Hypothesis.pdf"
QUERY="I want to understand more about the Linear Rep. Hypothesis, and about the casual inner product. Why did the Gemma work better than the LLama in this paper?"
MAX_ITER=3
OUTPUT_FORMAT="print"  # Options: "print" or "json"

echo "Running Query-Focused Summarization Demo..."
echo "Article: $ARTICLE_PATH"
echo "Query: $QUERY"
echo "Max Iterations: $MAX_ITER"
echo "Output Format: $OUTPUT_FORMAT"
echo "---"

source .venv/bin/activate && python3 src/main.py --file "$ARTICLE_PATH" --query "$QUERY" --max_iterations "$MAX_ITER" --output_format "$OUTPUT_FORMAT" > "./results/The Linear Representation Hypothesis.txt"

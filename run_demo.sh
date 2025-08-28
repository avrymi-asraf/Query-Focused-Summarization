#!/bin/bash
# Simple script to run the Query-Focused Summarization workflow

ARTICLE_PATH="articles/The-Linear-Representation-Hypothesis.md"
QUERY="Summarize the linear representation hypothesis and its implications for neural network interpretability."
MAX_ITER=5

echo "Running Query-Focused Summarization Demo..."
echo "Article: $ARTICLE_PATH"
echo "Query: $QUERY"
echo "Max Iterations: $MAX_ITER"
echo "---"

source .venv/bin/activate && python3 src/main.py --file "$ARTICLE_PATH" --query "$QUERY" --max_iterations "$MAX_ITER" --json_path "./results/The Linear Representation Hypothesis.json"

#!/usr/bin/env python3
"""
Simple script: send each Markdown file under 'articles' to Gemini with a fixed prompt.

Run:
  python scripts/generate_queries_from_md.py

Prerequisites:
  pip install google-genai
  export GEMINI_API_KEY=...
"""

import os
import sys
from dotenv import load_dotenv


PROMPT = """Find five different ways to summarize the article. The idea is to summarize how the article relates to a specific point that is not the main purpose of the article. Try to make it even a minor topic, but not too minor. That is, the article should have a lot of material on it.
It should be a difficult topic. So that it is difficult to summarize the article according to it.
Write it as a short quary. Not necessarily as a question sentence, even just as a topic"""


load_dotenv()

ARTICLES_DIR = os.path.abspath("articles")
MODEL = "gemini-2.5-pro"
MAX_CHARS = 120000

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
    raise SystemExit(1)

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception as exc:
    print(
        "[ERROR] google-genai is not installed. Install with: pip install google-genai\n"
        f"        Details: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)

if not os.path.isdir(ARTICLES_DIR):
    print(f"[ERROR] Folder not found: {ARTICLES_DIR}", file=sys.stderr)
    raise SystemExit(3)

client = genai.Client(api_key=api_key)

md_files = []
for current_dir, _subdirs, files in os.walk(ARTICLES_DIR):
    for name in files:
        if name.lower().endswith(".md"):
            md_files.append(os.path.join(current_dir, name))

if not md_files:
    print(f"No Markdown files found under: {ARTICLES_DIR}")
    raise SystemExit(0)

for path in sorted(md_files):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
        if len(data) > MAX_CHARS:
            data = data[:MAX_CHARS]

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=PROMPT),
                    types.Part.from_text(text=data),
                ],
            )
        ]

        resp = client.models.generate_content(model=MODEL, contents=contents)
        text = getattr(resp, "text", "").strip()

        print(f"\n=== {os.path.relpath(path, start=ARTICLES_DIR)} ===")
        print(text)
    except Exception as e:
        print(f"[WARN] Failed for {path}: {e}", file=sys.stderr)



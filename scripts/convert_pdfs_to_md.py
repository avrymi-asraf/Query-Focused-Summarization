#!/usr/bin/env python3
"""
Simple script: scan the 'articles' folder for PDFs and write Markdown next to them.

Run:
  python scripts/convert_pdfs_to_md.py

Prerequisite:
  pip install 'markitdown[all]'
"""

import os
import sys

ARTICLES_DIR = os.path.abspath("articles")

if not os.path.isdir(ARTICLES_DIR):
    print(f"[ERROR] Folder not found: {ARTICLES_DIR}", file=sys.stderr)
    raise SystemExit(1)

try:
    from markitdown import MarkItDown  # type: ignore
except Exception as exc:
    print(
        "[ERROR] markitdown is not installed. Install with: pip install 'markitdown[all]'\n"
        f"        Details: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)

md = MarkItDown(enable_plugins=False)

converted = 0
failed = 0

for current_dir, _subdirs, files in os.walk(ARTICLES_DIR):
    for name in files:
        if not name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(current_dir, name)
        md_path = os.path.splitext(pdf_path)[0] + ".md"
        try:
            result = md.convert(pdf_path)
            text = result.text_content or ""
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text)
            converted += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed to convert {pdf_path}: {e}", file=sys.stderr)

print(
    f"Done. Converted: {converted}, Failed: {failed}.\n"
    f"Scanned folder: {ARTICLES_DIR}"
)



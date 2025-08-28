# json to csv converter for our paper summarization results

# Usage example: python -m src.Json_to_csv --input results --output aggregated.csv
# Single file: python -m src.Json_to_csv --input results/sample.json --output single.csv
# Recursive: python -m src.Json_to_csv --input results --output aggregated.csv --recursive

# Use --input for the folder and --output for the single CSV.
#
# Basic (non‑recursive): python -m src.Json_to_csv --input results --output output\aggregated.csv
#
# Recursive through subfolders: python -m src.Json_to_csv --input results --output output\aggregated_recursive.csv --recursive
#
# Quiet (suppress warnings): python -m src.Json_to_csv --input results --output output\aggregated.csv --quiet

import json
import csv
import os
import argparse
import sys
from typing import List, Dict, Any, Iterable

PAPER_NAME_KEYS = ("paper_name", "title", "name")
SECTION_LIST_KEYS = (
    "sections_to_highlight",
    "sectionsHighlight",
    "sections_to_highlight_ids",
    "highlight_sections",
)

CSV_HEADERS = [
    "paper_name",
    "query",
    "max_iterations",
    "total_iterations_executed",
    "iteration_number",
    "questions_total",
    "questions_correct",
    "incorrect_questions",
    "accuracy",
    "acu_questions_total",
    "acu_questions_correct",
    "acu_accuracy",
    "judgment",
    "status",
    "sections_to_highlight_size",
    "source_file",
]

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_bool(val: Any) -> str:
    return "true" if bool(val) else "false"

def get_paper_name(root: Dict[str, Any], filename_stem: str) -> str:
    for k in PAPER_NAME_KEYS:
        v = root.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return filename_stem

def compute_acu_counts(root: Dict[str, Any],
                       judge_evals: List[Dict[str, Any]]) -> (int, int, bool):
    root_list = root.get("acu_questions") or []
    if isinstance(root_list, list) and root_list:
        acu_total = len(root_list)
        root_set = {q.strip() for q in root_list if isinstance(q, str)}
        acu_correct = sum(
            1 for ev in judge_evals
            if isinstance(ev.get("qa", {}), dict)
            and ev.get("qa", {}).get("question", "") in root_set
            and ev.get("result") is True
        )
        return acu_total, acu_correct, True
    prefix = [
        ev for ev in judge_evals
        if isinstance(ev.get("qa", {}), dict)
        and isinstance(ev.get("qa", {}).get("question", ""), str)
        and ev.get("qa", {}).get("question", "").startswith("ACU.")
    ]
    if prefix:
        acu_total = len(prefix)
        acu_correct = sum(1 for ev in prefix if ev.get("result") is True)
        return acu_total, acu_correct, True
    return 0, 0, False

def safe_div(num: int, den: int) -> str:
    return "" if den == 0 else f"{num/den:.4f}"

def derive_size_from_lists(containers: List[Dict[str, Any]]) -> int:
    for c in containers:
        if not isinstance(c, dict):
            continue
        for key in SECTION_LIST_KEYS:
            if key in c:
                val = c.get(key)
                if isinstance(val, list):
                    return len(val)
                return 0
    return -1

def get_sections_to_highlight_size(root: Dict[str, Any],
                                   iteration: Dict[str, Any],
                                   judge: Dict[str, Any]) -> int:
    for scope in (iteration, judge, root):
        if isinstance(scope, dict) and "sections_to_highlight_size" in scope:
            v = scope.get("sections_to_highlight_size")
            if isinstance(v, int) and v >= 0:
                return v
    # Derive from any list in iteration -> judge -> root
    return derive_size_from_lists([iteration, judge, root])

def extract_rows(root: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    filename_stem = os.path.splitext(os.path.basename(source_file))[0]
    paper_name = get_paper_name(root, filename_stem)

    iterations = root.get("iterations") or []
    total_iterations_executed = root.get("total_iterations")
    if total_iterations_executed is None:
        total_iterations_executed = len(iterations)

    rows: List[Dict[str, Any]] = []
    for iteration in iterations:
        judge = iteration.get("judge") or {}
        judge_evals = judge.get("evaluations") or []
        questions_total = len(judge_evals)
        questions_correct = sum(1 for ev in judge_evals if ev.get("result") is True)
        acu_total, acu_correct, acu_present = compute_acu_counts(root, judge_evals)
        sections_size = get_sections_to_highlight_size(root, iteration, judge)

        rows.append({
            "paper_name": paper_name,
            "query": root.get("query", ""),
            "max_iterations": root.get("max_iterations", ""),
            "total_iterations_executed": total_iterations_executed,
            "iteration_number": iteration.get("iteration_number", ""),
            "questions_total": questions_total,
            "questions_correct": questions_correct,
            "incorrect_questions": (questions_total - questions_correct) if questions_total else 0,
            "accuracy": safe_div(questions_correct, questions_total),
            "acu_questions_total": acu_total if acu_present else "",
            "acu_questions_correct": acu_correct if acu_present else "",
            "acu_accuracy": safe_div(acu_correct, acu_total) if (acu_present and acu_total > 0) else "",
            "judgment": normalize_bool(judge.get("judgment")) if "judgment" in judge else "",
            "status": root.get("status", ""),
            "sections_to_highlight_size": sections_size,
            "source_file": os.path.basename(source_file),
        })
    return rows

def iter_json_files(path: str, recursive: bool) -> Iterable[str]:
    if os.path.isfile(path):
        if path.lower().endswith(".json"):
            yield path
        return
    for root_dir, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".json"):
                yield os.path.join(root_dir, f)
        if not recursive:
            break

def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Convert summarization JSON result files to a CSV.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to a JSON file or a directory containing JSON files.")
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV file path.")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recursively search subdirectories for JSON files when input is a directory.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress warnings.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input path not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []
    files_processed = 0
    for jf in iter_json_files(args.input, args.recursive):
        try:
            data = load_json(jf)
            all_rows.extend(extract_rows(data, jf))
            files_processed += 1
        except Exception as e:
            if not args.quiet:
                print(f"Warning: failed processing {jf}: {e}", file=sys.stderr)

    if files_processed == 0:
        print("No JSON files processed.", file=sys.stderr)
        sys.exit(2)

    write_csv(all_rows, args.output)
    if not args.quiet:
        print(f"Wrote {len(all_rows)} rows from {files_processed} file(s) to {args.output}")

if __name__ == "__main__":
    main()


# PAPER_NAME_KEYS = ("paper_name", "title", "name")
#
# CSV_HEADERS = [
#     "paper_name",
#     "query",
#     "max_iterations",
#     "total_iterations_executed",
#     "iteration_number",
#     "questions_total",
#     "questions_correct",
#     "incorrect_questions",
#     "accuracy",
#     "acu_questions_total",
#     "acu_questions_correct",
#     "acu_accuracy",
#     "judgment",
#     "status",
#     "source_file",
# ]
#
# def load_json(path: str) -> Any:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
# def normalize_bool(val: Any) -> str:
#     return "true" if bool(val) else "false"
#
# def get_paper_name(root: Dict[str, Any], filename_stem: str) -> str:
#     for k in PAPER_NAME_KEYS:
#         v = root.get(k)
#         if isinstance(v, str) and v.strip():
#             return v.strip()
#     return filename_stem
#
# def compute_acu_counts(root: Dict[str, Any],
#                        judge_evals: List[Dict[str, Any]]) -> (int, int, bool):
#     # Returns (acu_total, acu_correct, present_flag)
#     root_list = root.get("acu_questions") or []
#     if isinstance(root_list, list) and len(root_list) > 0:
#         acu_total = len(root_list)
#         # Need to match by prefix OR membership? We only have questions text; count correct among judge evals whose question matches any root acu question text
#         root_set = set(q.strip() for q in root_list if isinstance(q, str))
#         acu_correct = 0
#         for ev in judge_evals:
#             qa = ev.get("qa", {}) or {}
#             qtext = qa.get("question", "")
#             if qtext in root_set and ev.get("result") is True:
#                 acu_correct += 1
#         return acu_total, acu_correct, True
#     # Fallback to ACU. prefix
#     prefix_evals = []
#     for ev in judge_evals:
#         qa = ev.get("qa", {}) or {}
#         qtext = qa.get("question", "")
#         if isinstance(qtext, str) and qtext.startswith("ACU."):
#             prefix_evals.append(ev)
#     if prefix_evals:
#         acu_total = len(prefix_evals)
#         acu_correct = sum(1 for ev in prefix_evals if ev.get("result") is True)
#         return acu_total, acu_correct, True
#     return 0, 0, False
#
# def safe_div(num: int, den: int) -> str:
#     if den == 0:
#         return ""
#     return f"{num/den:.4f}"
#
# def extract_rows(root: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
#     filename_stem = os.path.splitext(os.path.basename(source_file))[0]
#     paper_name = get_paper_name(root, filename_stem)
#
#     iterations = root.get("iterations") or []
#     total_iterations_executed = root.get("total_iterations")
#     if total_iterations_executed is None:
#         total_iterations_executed = len(iterations)
#
#     rows: List[Dict[str, Any]] = []
#     for iteration in iterations:
#         judge = iteration.get("judge") or {}
#         judge_evals = judge.get("evaluations") or []
#         questions_total = len(judge_evals)
#         questions_correct = sum(1 for ev in judge_evals if ev.get("result") is True)
#
#         acu_total, acu_correct, acu_present = compute_acu_counts(root, judge_evals)
#
#         row = {
#             "paper_name": paper_name,
#             "query": root.get("query", ""),
#             "max_iterations": root.get("max_iterations", ""),
#             "total_iterations_executed": total_iterations_executed,
#             "iteration_number": iteration.get("iteration_number", ""),
#             "questions_total": questions_total,
#             "questions_correct": questions_correct,
#             "incorrect_questions": (questions_total - questions_correct) if questions_total else 0,
#             "accuracy": safe_div(questions_correct, questions_total),
#             "acu_questions_total": acu_total if acu_present else "",
#             "acu_questions_correct": acu_correct if acu_present else "",
#             "acu_accuracy": safe_div(acu_correct, acu_total) if (acu_present and acu_total > 0) else "",
#             "judgment": normalize_bool(judge.get("judgment")) if "judgment" in judge else "",
#             "status": root.get("status", ""),
#             "source_file": os.path.basename(source_file),
#         }
#         rows.append(row)
#     return rows
#
# def iter_json_files(path: str, recursive: bool) -> Iterable[str]:
#     if os.path.isfile(path):
#         if path.lower().endswith(".json"):
#             yield path
#         return
#     for root, dirs, files in os.walk(path):
#         for f in files:
#             if f.lower().endswith(".json"):
#                 yield os.path.join(root, f)
#         if not recursive:
#             break
#
# def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     with open(out_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
#         writer.writeheader()
#         for r in rows:
#             writer.writerow(r)
#
# def main():
#     parser = argparse.ArgumentParser(description="Convert summarization JSON result files to a CSV.")
#     parser.add_argument("--input", "-i", required=True,
#                         help="Path to a JSON file or a directory containing JSON files.")
#     parser.add_argument("--output", "-o", required=True,
#                         help="Output CSV file path.")
#     parser.add_argument("--recursive", "-r", action="store_true",
#                         help="Recursively search subdirectories for JSON files when input is a directory.")
#     parser.add_argument("--quiet", "-q", action="store_true", help="Suppress warnings.")
#     args = parser.parse_args()
#
#     input_path = args.input
#     if not os.path.exists(input_path):
#         print(f"Error: input path not found: {input_path}", file=sys.stderr)
#         sys.exit(1)
#
#     all_rows: List[Dict[str, Any]] = []
#     files_processed = 0
#     for jf in iter_json_files(input_path, args.recursive):
#         try:
#             data = load_json(jf)
#             rows = extract_rows(data, jf)
#             all_rows.extend(rows)
#             files_processed += 1
#         except Exception as e:
#             if not args.quiet:
#                 print(f"Warning: failed processing {jf}: {e}", file=sys.stderr)
#
#     if files_processed == 0:
#         print("No JSON files processed.", file=sys.stderr)
#         sys.exit(2)
#
#     write_csv(all_rows, args.output)
#     if not args.quiet:
#         print(f"Wrote {len(all_rows)} rows from {files_processed} file(s) to {args.output}")
#
# if __name__ == "__main__":
#     main()
#
# Use --input for the folder and --output for the single CSV.
#
# Basic (non‑recursive): python -m src.Json_to_csv --input results --output output\aggregated.csv
#
# Recursive through subfolders: python -m src.Json_to_csv --input results --output output\aggregated_recursive.csv --recursive
#
# Quiet (suppress warnings): python -m src.Json_to_csv --input results --output output\aggregated.csv --quiet
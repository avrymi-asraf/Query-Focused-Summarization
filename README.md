# Query-Focused Summarization

Query-Focused Summarization is a small LangChain workflow for producing summaries that answer a specific user question instead of producing a generic article overview. It reads a text or PDF article, asks an LLM to generate diagnostic questions, creates a query-focused summary, evaluates whether the summary answers those questions, and iterates until a judge agent is satisfied or the iteration limit is reached.

The repository is intentionally compact, so it is useful both as a runnable summarization tool and as a reference implementation for agentic evaluation loops.

## Table of contents

- [What this project does](#what-this-project-does)
- [Repository structure](#repository-structure)
- [Core ideas and techniques](#core-ideas-and-techniques)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [End-to-end workflow](#end-to-end-workflow)
- [Using the tools from Python](#using-the-tools-from-python)
- [Agent and parser reference](#agent-and-parser-reference)
- [Input file handling](#input-file-handling)
- [Output formats](#output-formats)
- [Practical examples](#practical-examples)
- [How to adapt or extend the project](#how-to-adapt-or-extend-the-project)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## What this project does

Most summarizers answer the broad prompt, “summarize this document.” This project answers a narrower prompt: “summarize this document in a way that helps with this query.” The query is carried through the workflow so each stage can judge relevance against the user’s actual information need.

For example, given a long article about Hebrew University and the query:

```text
What are the main research areas and academic strengths?
```

The workflow should ignore unrelated details where possible and prioritize material about faculties, campuses, libraries, research institutes, rankings, and disciplinary strengths.

## Repository structure

```text
.
├── README.md                    # This guide
├── requirements.txt             # Python dependencies
├── articals/
│   └── hebrew-university.md     # Example article used in sample commands
└── src/
    ├── Agents.py                # LLM setup, output parsers, and agent classes
    └── main.py                  # CLI, file loading, PDF handling, workflow orchestration
```

> Note: the example content directory is currently named `articals/` in the repository, so commands should use that spelling.

## Core ideas and techniques

### 1. Query-focused summarization

The summary prompt receives both the article and the user query. This keeps the model from spending limited output tokens on content that may be true but irrelevant.

The summarizer is instructed to produce two sections:

1. `SUMMARY`: a cohesive 200–250 word overview that directly addresses the query.
2. `KEY HIGHLIGHTS`: 3–5 concise statements with the most important facts, data points, or claims.

The relevant implementation is the `Summarizer` class, which injects `{query}`, `{article}`, and `{sections}` into a LangChain chat prompt before parsing the LLM response as a string.

```python
class Summarizer:
    def __init__(self, llm=None):
        self.llm = llm or _llm_summarizer
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", "... User's Query/Perspective:\n{query}\n\n ...")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query: str, article: str, sections: List[str]) -> str:
        return self.chain.invoke({"query": query, "article": article, "sections": "\n".join(sections)})
```

### 2. Diagnostic questions

Before the first summary is written, the `QuestionGenerator` creates exactly five diagnostic questions. These questions act like a coverage checklist for the rest of the workflow.

The prompt asks for a mix of:

- factual questions,
- analytical questions,
- inferential questions,
- questions covering different article sections,
- questions that are answerable from the article and relevant to the query.

This technique makes the workflow less dependent on a single summarization pass. If the summary cannot answer the diagnostic questions, the system has evidence that it missed something important.

### 3. Self-evaluation with QA pairs

The `QAAgent` answers the diagnostic questions using only the generated summary. It is not allowed to use the original article or outside knowledge. If the summary does not contain enough information, it must answer exactly:

```text
Not enough information in summary
```

This creates a useful separation of concerns:

- The summarizer writes the best answer it can.
- The QA agent tests whether the answer contains enough information.
- The judge compares the summary, QA pairs, and article to decide whether another pass is needed.

### 4. Judge-driven iteration

The `Judge` receives the original article, current summary, and QA pairs. It evaluates:

1. factual accuracy,
2. completeness,
3. specificity,
4. QA accuracy.

If the summary is good enough, the judge returns exactly `OK`. Otherwise, it returns missing or incorrectly handled topics, one per line. Those topics become the `sections` focus list for the next summarization pass.

This is a lightweight “reflection loop”:

```text
article + query
   ↓
generate diagnostic questions
   ↓
summarize with current focus topics
   ↓
answer diagnostic questions from summary only
   ↓
judge against article
   ↓
OK? return final summary
   ↓
missing topics? summarize again with those topics emphasized
```

### 5. LangChain Expression Language (LCEL)

The project uses LangChain’s pipe syntax to build chains. A typical chain is:

```python
self.chain = self.prompt | self.llm | StrOutputParser() | QuestionListParser()
```

Read this left to right:

1. `self.prompt` formats a prompt from input variables.
2. `self.llm` sends the prompt to Gemini through LangChain.
3. `StrOutputParser()` converts the model response to plain text.
4. A custom parser converts text into the desired Python structure.

This style keeps each agent small and composable.

### 6. Rate limiting

The repository configures an in-memory LangChain rate limiter before creating the model clients:

```python
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.233,
    check_every_n_seconds=0.1,
    max_bucket_size=14,
)
```

That rate limiter is passed to both model instances. This is useful when working with free-tier or quota-limited LLM APIs because the workflow can make several model calls per iteration:

- one call for question generation,
- one call for summarization per iteration,
- one call for QA per iteration,
- one call for judging per iteration.

With `max_iterations=3`, a full run can make up to `1 + (3 × 3) = 10` LLM calls.

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The dependency file includes:

- `python-dotenv` for loading `.env` variables,
- `langchain-core` for prompts, chains, output parsers, and rate limiting,
- `langchain-google-genai` for Gemini chat models,
- `langchain-community` for document loaders,
- `pypdf` and `unstructured[pdf]` for PDF extraction.

## Configuration

The agents call Google Gemini through `langchain-google-genai`, so you need a Google API key available in your environment.

Create a `.env` file in the repository root:

```bash
cat > .env <<'EOF_ENV'
GOOGLE_API_KEY=your-google-api-key-here
EOF_ENV
```

The code calls `load_dotenv()` when `src/Agents.py` is imported, so values in `.env` are loaded automatically.

The default model is configured in `src/Agents.py`:

```python
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", rate_limiter=rate_limiter)
_llm_summarizer = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_output_tokens=400, rate_limiter=rate_limiter)
```

The summarizer model is given `max_output_tokens=400` to keep summaries concise. If your summaries are being cut off, increase that value.

## Quick start

Run the included example article:

```bash
python src/main.py \
  --file articals/hebrew-university.md \
  --query "What are the main research areas?" \
  --max_iterations 3
```

For structured output:

```bash
python src/main.py \
  --file articals/hebrew-university.md \
  --query "What are the main research areas?" \
  --max_iterations 3 \
  --output_format json
```

Write JSON directly to a file:

```bash
python src/main.py \
  --file articals/hebrew-university.md \
  --query "What are the university's academic strengths, campuses, and research resources?" \
  --max_iterations 3 \
  --output_format json \
  --json_path outputs/hebrew-university-summary.json
```

## Command-line usage

The CLI lives in `src/main.py` and exposes these arguments:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--file` | Yes | none | Path to a text, Markdown, or PDF article. |
| `--query` | Yes | none | The user question or perspective for the summary. |
| `--max_iterations` | No | `5` in the CLI | Maximum number of summarize → QA → judge loops. |
| `--output_format` | No | `print` | Either `print` for human-readable console output or `json` for structured data. |
| `--json_path` | No | none | When using JSON output, write the structured result to this file instead of stdout. |

Display the built-in help:

```bash
python src/main.py --help
```

## End-to-end workflow

The central orchestration function is `run_summarization_workflow` in `src/main.py`.

```python
def run_summarization_workflow(query: str, article: str, max_iterations: int = 4, output_format: str = "print"):
    question_gen = QuestionGenerator()
    summarizer = Summarizer()
    qa_agent = QAAgent()
    judge_agent = Judge()

    questions = question_gen.run(query=query, article=article)
    current_summary = ""
    sections_to_highlight = []
```

The function then loops up to `max_iterations` times:

```python
for iteration in range(max_iterations):
    current_summary = summarizer.run(query=query, article=article, sections=sections_to_highlight)
    qa_pairs = qa_agent.run(questions=questions, summary=current_summary)
    needs_iteration, missing_topics = judge_agent.run(
        article=article,
        summary=current_summary,
        qa_pairs=qa_pairs,
    )
```

If the judge returns `OK`, the workflow returns immediately. If not, the missing topics are passed into the next summary call:

```python
sections_to_highlight = missing_topics
```

That is the core feedback mechanism. The next summary is not blind; it is told exactly what the judge found missing.

## Using the tools from Python

You can import and run the workflow directly instead of using the CLI.

```python
from src.main import load_file_content, run_summarization_workflow

article = load_file_content("articals/hebrew-university.md")

result = run_summarization_workflow(
    query="What are the main research areas and academic strengths?",
    article=article,
    max_iterations=3,
    output_format="json",
)

print(result["final_summary"])
print(result["status"])
```

You can also use individual agents when experimenting with prompts or evaluation.

```python
from src.Agents import QuestionGenerator, Summarizer, QAAgent, Judge

article = "Hebrew University has faculties in science, medicine, agriculture, law, humanities, and social sciences."
query = "Which academic areas are represented?"

question_gen = QuestionGenerator()
summarizer = Summarizer()
qa_agent = QAAgent()
judge = Judge()

questions = question_gen.run(query=query, article=article)
summary = summarizer.run(query=query, article=article, sections=[])
qa_pairs = qa_agent.run(questions=questions, summary=summary)
needs_iteration, missing_topics = judge.run(article=article, summary=summary, qa_pairs=qa_pairs)

print(questions)
print(summary)
print(qa_pairs)
print(needs_iteration, missing_topics)
```

### Dependency-injecting a custom LLM

Each agent accepts an optional `llm` argument. This is useful for testing, swapping providers, or using different model settings.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from src.Agents import Summarizer

custom_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_output_tokens=800,
)

summarizer = Summarizer(llm=custom_llm)
summary = summarizer.run(
    query="What does the article say about research infrastructure?",
    article="...",
    sections=["libraries", "archives", "research centers"],
)
```

## Agent and parser reference

### `QuestionGenerator`

**Purpose:** Create five diagnostic questions from the query and article.

**Input:**

```python
run(query: str, article: str) -> List[str]
```

**Output example:**

```python
[
    "Which faculties or institutes are associated with scientific research?",
    "What research resources does the university provide through its libraries or archives?",
    "How do rankings support the article's claims about academic strengths?",
    "Which campuses host medicine, science, agriculture, or social science programs?",
    "What evidence does the article give for international academic activity?",
]
```

**How it works:** The prompt asks for exactly five unnumbered questions. `QuestionListParser` then splits the model output on newlines and removes blank lines.

### `Summarizer`

**Purpose:** Write a concise answer to the user query using the article.

**Input:**

```python
run(query: str, article: str, sections: List[str]) -> str
```

**Key behavior:**

- The `query` defines relevance.
- The full `article` supplies evidence.
- The `sections` list contains missing topics from a previous judge pass. On the first iteration, it is empty.

**Output example:**

```text
1. SUMMARY: ...

2. KEY HIGHLIGHTS:
- ...
- ...
- ...
```

### `QAAgent`

**Purpose:** Check whether the summary can answer the diagnostic questions.

**Input:**

```python
run(questions: List[str], summary: str) -> List[Tuple[str, str]]
```

**Output example:**

```python
[
    ("Which campuses host science programs?", "The Givat Ram campus hosts the Faculty of Science."),
    ("Which campus hosts medicine?", "The Ein Kerem campus hosts Medicine and Dental Medicine."),
]
```

**Important technique:** It is told to use only the summary. This makes missing information visible instead of letting the model fill gaps from the article or outside knowledge.

### `Judge`

**Purpose:** Decide whether the current summary is complete and accurate enough.

**Input:**

```python
run(article: str, summary: str, qa_pairs: List[Tuple[str, str]]) -> Tuple[bool, List[str]]
```

**Output examples:**

If satisfied:

```python
(False, [])
```

If not satisfied:

```python
(True, [
    "Include the role of the Jewish National and University Library",
    "Add specific campus-to-faculty mappings",
    "Mention ranking evidence for academic strength",
])
```

### Custom parsers

The project defines three `BaseOutputParser` subclasses:

| Parser | Converts | Used by |
| --- | --- | --- |
| `QuestionListParser` | newline-separated model text → `List[str]` | `QuestionGenerator` |
| `QAPairsParser` | `question: answer` lines → `List[Tuple[str, str]]` | `QAAgent` |
| `JudgeOutputParser` | `OK` or topic lines → `(needs_iteration, missing_topics)` | `Judge` |

These parsers are deliberately simple. If you need stronger guarantees, consider asking the model for JSON and parsing with a JSON parser or Pydantic schema.

## Input file handling

The `load_file_content` helper chooses between direct text reading and PDF extraction.

```python
def load_file_content(file_path: str) -> str:
    if file_path.lower().endswith('.pdf'):
        print(f"Processing PDF file: {file_path}")
        return process_pdf_to_markdown(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
```

### Text and Markdown files

Text-like files are read as UTF-8 and passed directly to the workflow. This is the simplest path and works well for `.txt` and `.md` files.

### PDF files

PDFs are converted to Markdown-style text before summarization:

1. `PyPDFLoader` is tried first.
2. If `PyPDFLoader` fails, `UnstructuredPDFLoader` is used as a fallback.
3. Extracted content is wrapped in page or section headings such as `# Page 1` or `# Section 1`.

This gives the model weak structural cues about where content came from.

## Output formats

### Human-readable console output

The default `print` mode prints:

- iteration headings,
- the generated summary for each iteration,
- QA pairs for each iteration,
- judge status,
- final summary,
- total iterations.

Use this mode while developing prompts because it shows the loop’s reasoning artifacts.

```bash
python src/main.py \
  --file articals/hebrew-university.md \
  --query "What does the article say about libraries and archives?"
```

### JSON output

JSON mode returns structured data:

```json
{
  "query": "What does the article say about libraries and archives?",
  "max_iterations": 3,
  "iterations": [
    {
      "iteration_number": 1,
      "summary": "...",
      "qa_pairs": [["Question", "Answer"]],
      "needs_iteration": true,
      "missing_topics": ["..."]
    }
  ],
  "final_summary": "...",
  "total_iterations": 2,
  "status": "completed"
}
```

Use JSON mode when integrating this project into scripts, notebooks, benchmarks, or web services.

## Practical examples

### Example 1: Summarize academic strengths

```bash
python src/main.py \
  --file articals/hebrew-university.md \
  --query "What are Hebrew University's main academic strengths and research areas?" \
  --max_iterations 3
```

Good queries are specific. This one asks for academic strengths and research areas, so the summary should focus on faculties, institutes, libraries, archives, and rankings.

### Example 2: Focus on campuses

```bash
python src/main.py \
  --file articals/hebrew-university.md \
  --query "Which campuses are mentioned, and what academic units are located on each campus?" \
  --max_iterations 2
```

This query should make the summary organize information by campus, such as Mount Scopus, Edmond J. Safra/Givat Ram, Ein Kerem, and Rehovot.

### Example 3: Produce machine-readable benchmark data

```bash
mkdir -p outputs
python src/main.py \
  --file articals/hebrew-university.md \
  --query "What evidence does the article give for the university's research reputation?" \
  --max_iterations 4 \
  --output_format json \
  --json_path outputs/research-reputation.json
```

This is useful when comparing different prompts, models, or iteration counts. The JSON records each iteration, including what the judge considered missing.

### Example 4: Summarize a PDF

```bash
python src/main.py \
  --file papers/example-paper.pdf \
  --query "What methods and evaluation metrics does this paper use?" \
  --max_iterations 3 \
  --output_format json
```

PDF extraction quality varies. If a PDF has poor text extraction, try converting it to Markdown or plain text externally before using this workflow.

### Example 5: Batch over many queries

```python
from src.main import load_file_content, run_summarization_workflow

article = load_file_content("articals/hebrew-university.md")
queries = [
    "What campuses and faculties are described?",
    "What library and archive resources are mentioned?",
    "What rankings or reputation signals are included?",
]

for query in queries:
    result = run_summarization_workflow(
        query=query,
        article=article,
        max_iterations=2,
        output_format="json",
    )
    print("\nQUERY:", query)
    print("STATUS:", result["status"])
    print(result["final_summary"])
```

## How to adapt or extend the project

### Change the model

Edit the model declarations in `src/Agents.py`:

```python
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", rate_limiter=rate_limiter)
_llm_summarizer = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_output_tokens=400, rate_limiter=rate_limiter)
```

You might use:

- a faster model for question generation and QA,
- a larger-context model for long PDFs,
- a higher-output-token summarizer for detailed reports,
- `temperature=0` for more deterministic outputs.

### Improve parser robustness

The current parsers assume newline-separated questions, colon-separated QA pairs, and exact `OK` for judge success. This keeps the code simple but can be brittle.

Options:

- Ask the LLM to emit JSON.
- Add validation and retries when parsing fails.
- Strip numbering from question lines.
- Preserve multiline answers in QA parsing.
- Use Pydantic output parsers for typed schemas.

### Add chunking for very long documents

Currently, the full article is sent to each relevant agent call. For long documents, you may hit context limits or spend unnecessary tokens.

Common approaches:

1. Split the article into chunks.
2. Retrieve chunks most relevant to the query.
3. Summarize retrieved chunks.
4. Feed chunk summaries into the existing judge loop.

A simple extension point is `load_file_content`: after loading text, split or preprocess it before calling `run_summarization_workflow`.

### Add citations to summaries

The PDF loader already adds page headings. You can modify the summarizer prompt to require references like `[Page 3]` whenever it uses a fact. For Markdown documents, you can preserve section headings and ask the model to cite them.

### Add tests with fake LLMs

The agent constructors accept custom `llm` instances, so tests can inject deterministic fake models. This avoids live API calls in unit tests.

A fake LLM can be used to check:

- parser behavior,
- workflow status transitions,
- handling of `OK` vs missing topics,
- JSON result structure.

### Make the CLI more production-friendly

Potential improvements:

- add `--model`, `--temperature`, and `--max_output_tokens` arguments,
- support `.docx`, `.html`, or URL inputs,
- add logging instead of `print`,
- write intermediate artifacts to disk,
- expose the workflow through an API server,
- add retry/backoff for transient API failures.

## Troubleshooting

### `GOOGLE_API_KEY` errors

Make sure your `.env` file exists and contains:

```text
GOOGLE_API_KEY=your-google-api-key-here
```

Also make sure you run commands from the repository root so `load_dotenv()` can find the file.

### PDF loading fails

Try these steps:

1. Confirm the PDF contains selectable text rather than only scanned images.
2. Install optional PDF dependencies from `unstructured[pdf]`.
3. Convert the PDF to text or Markdown manually and pass that file instead.
4. Test with a small PDF first.

### Summaries are too short or incomplete

Try:

- increasing `--max_iterations`,
- increasing `max_output_tokens` for `_llm_summarizer`,
- making the query more specific,
- changing the summarizer prompt from 200–250 words to a larger target,
- switching to JSON output to inspect missing topics across iterations.

### The judge keeps asking for another iteration

This can happen when the prompt asks for more detail than the output token budget allows. Increase summary length or token budget, or make the judge criteria less strict.

### QA parsing loses information

`QAPairsParser` splits each line at the first colon. If the model emits multiline answers or omits colons, data may be dropped. For production use, prefer JSON output from the model.

## Roadmap

- [x] Main CLI with arguments
- [x] Basic logging/console progress output
- [x] Text and PDF loading helpers
- [x] Comprehensive usage guide
- [ ] Tests for basic workflow behavior
- [ ] Benchmarks for performance and summary quality
- [ ] Support additional file types
- [ ] Store results for every branch or experiment outside the codebase
- [ ] Add interactive mode for requesting more iterations or adding focus topics

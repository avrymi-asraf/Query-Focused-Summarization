from Agents import (
    QuestionGenerator,
    Summarizer,
    QAAgentRunner,
    Judge,
    JudgeEvaluationType,
    QuestionsOutputType,
    QAAgentEvaluationsOutputType,
)
import argparse
import os
import json
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# --- Add/replace in src/main.py ---

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

FALLBACK_ANSWERS = {
    "not enough information in summary",
    "not enough information",
}

def process_pdf_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to markdown text.
    Tries PyPDFLoader first, falls back to UnstructuredPDFLoader.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        parts = []
        for i, doc in enumerate(documents, start=1):
            content = (doc.page_content or "").strip()
            if content:
                parts.append(f"# Page {i}\n\n{content}\n")
        md = "\n".join(parts).strip()
        if not md:
            raise ValueError("Empty content extracted by PyPDFLoader.")
        return md
    except Exception as primary_err:
        print(f"PyPDFLoader failed: {primary_err}. Trying UnstructuredPDFLoader...")
        try:
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            parts = []
            for i, doc in enumerate(documents, start=1):
                content = (doc.page_content or "").strip()
                if content:
                    parts.append(f"# Section {i}\n\n{content}\n")
            md = "\n".join(parts).strip()
            if not md:
                raise ValueError("Empty content extracted by UnstructuredPDFLoader.")
            return md
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both PDF loaders failed. Primary: {primary_err}; Fallback: {fallback_err}"
            ) from fallback_err

def load_file_content(file_path: str) -> str:
    """
    Load file content. If PDF, convert to markdown first.
    """
    if file_path.lower().endswith(".pdf"):
        print(f"Processing PDF file: {file_path}")
        content = process_pdf_to_markdown(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    if not content or not content.strip():
        raise ValueError("Loaded article content is empty.")
    return content

def run_summarization_workflow(
    query: str,
    article: str,
    max_iterations: int = 4,
    requests_per_second: float | None = None,
):
    """
    Run iterative summarization workflow with added QA failure gating.
    """
    if not article or not article.strip():
        raise ValueError("Article content is empty. Aborting workflow.")

    question_gen = QuestionGenerator(requests_per_second=requests_per_second)
    summarizer = Summarizer(requests_per_second=requests_per_second)
    qa_agent = QAAgentRunner(requests_per_second=requests_per_second)
    judge_agent = Judge(requests_per_second=requests_per_second)

    questions_output: QuestionsOutputType = question_gen.run(query=query, article=article)
    questions = questions_output.questions
    acu_questions = getattr(questions_output, "acu_questions", [])
    current_summary = ""
    sections_to_highlight: list[str] = []

    workflow_result = {
        "query": query,
        "max_iterations": max_iterations,
        "iterations": [],
        "final_summary": "",
        "total_iterations": 0,
        "status": "",
        "questions": questions,
        "acu_questions": acu_questions,
    }

    for iteration in range(max_iterations):
        iteration_data = {
            "iteration_number": iteration + 1,
            "summary": "",
            "qa_evaluations": [],
            "qa_pairs": [],
            "judge": None,
            "correct_count_all": 0,
            "correct_count_acu": 0,
            "num_of_questions": len(questions),
            "sections_to_highlight": [],
            "sections_to_highlight_size": 0,
            "qa_failures_present": False,
        }

        # Summarize
        current_summary = summarizer.run(
            query=query, article=article, sections=list(sections_to_highlight)
        )
        iteration_data["summary"] = current_summary

        # QA (summary-only)
        qa_evaluations_struct: QAAgentEvaluationsOutputType = qa_agent.run(
            questions_output=questions_output, summary=current_summary
        )

        # Serialize evaluations
        iteration_data["qa_evaluations"] = [
            ev.model_dump() for ev in qa_evaluations_struct.evaluations
        ]

        # Prepare QA pairs and detect failures
        qa_pairs = []
        qa_failures = False
        for ev in qa_evaluations_struct.evaluations:
            ans_text = ev.qa.answer
            qa_pairs.append({"question": ev.qa.question, "answer": ans_text})
            if not ev.result:
                qa_failures = True
        iteration_data["qa_pairs"] = qa_pairs
        iteration_data["qa_failures_present"] = qa_failures

        # If QA answers are all good → summary is good → finish without Judge
        if not qa_failures:
            total_correct_qa = sum(1 for ev in qa_evaluations_struct.evaluations if ev.result is True)
            acu_correct_qa = sum(
                1
                for ev in qa_evaluations_struct.evaluations
                if ev.result is True
                and isinstance(ev.qa.question, str)
                and ev.qa.question.strip().startswith("ACU.")
            )
            iteration_data["correct_count_all"] = total_correct_qa
            iteration_data["correct_count_acu"] = acu_correct_qa
            iteration_data["sections_to_highlight"] = list(sections_to_highlight)
            iteration_data["sections_to_highlight_size"] = len(iteration_data["sections_to_highlight"]) if iteration_data["sections_to_highlight"] is not None else 0
            workflow_result["iterations"].append(iteration_data)
            workflow_result["final_summary"] = current_summary
            workflow_result["total_iterations"] = iteration + 1
            workflow_result["status"] = "completed"
            return workflow_result

        # Otherwise, call Judge to update sections_to_highlight using conversation context
        # Build brief conversation context from previous iterations and current QA issues
        previous_sections = []
        seen = set()
        for it in workflow_result["iterations"]:
            for s in it.get("sections_to_highlight", []) or []:
                if s not in seen:
                    previous_sections.append(s)
                    seen.add(s)
        failed_qa_notes = []
        for ev in qa_evaluations_struct.evaluations:
            if not ev.result:
                q = ev.qa.question
                issue = ev.issue or "Not enough information"
                failed_qa_notes.append(f"- {q} — {issue}")
        conversation_notes = (
            f"Query: {query}\n"
            f"Iterations so far: {len(workflow_result['iterations'])}\n"
            + ("Previously proposed sections (ordered):\n" + "\n".join(f"- {s}" for s in previous_sections) + "\n" if previous_sections else "")
            + ("Current QA issues: \n" + "\n".join(failed_qa_notes) if failed_qa_notes else "")
        ).strip()

        judge_eval: JudgeEvaluationType = judge_agent.run(
            article=article,
            summary=current_summary,
            qa_pairs=qa_pairs,
            sections_context=list(sections_to_highlight),
            conversation_notes=conversation_notes,
        )

        iteration_data["judge"] = judge_eval.model_dump()
        iteration_data["sections_to_highlight"] = list(sections_to_highlight)
        iteration_data["sections_to_highlight_size"] = len(sections_to_highlight)

        total_correct = sum(1 for ev in judge_eval.evaluations if ev.result is True)
        acu_correct = sum(
            1
            for ev in judge_eval.evaluations
            if ev.result is True
            and isinstance(ev.qa.question, str)
            and ev.qa.question.strip().startswith("ACU.")
        )
        iteration_data["correct_count_all"] = total_correct
        iteration_data["correct_count_acu"] = acu_correct

        workflow_result["iterations"].append(iteration_data)

        # NEW stop condition: require judge true AND no QAAgent failures
        if judge_eval.judgment and not qa_failures:
            workflow_result["final_summary"] = current_summary
            workflow_result["total_iterations"] = iteration + 1
            workflow_result["status"] = "completed"
            return workflow_result
        else:
            # Accumulate sections across iterations using a set for O(1) membership checks
            new_sections = list(getattr(judge_eval, "sections_to_highlight", []))
            for section in new_sections:
                sections_to_highlight.add(section)
            print(f"sections_to_highlight: {sections_to_highlight}, iteration: {iteration}")

    workflow_result["final_summary"] = current_summary
    workflow_result["total_iterations"] = max_iterations
    workflow_result["status"] = "max_iterations_reached"
    return workflow_result



# from Agents import (
#     QuestionGenerator,
#     Summarizer,
#     QAAgentRunner,
#     Judge,
#     JudgeEvaluationType,
#     QuestionsOutputType,
#     QAAgentEvaluationsOutputType,
# )
# import argparse
# import os
# import json
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
#
# def process_pdf_to_markdown(file_path: str) -> str:
#     """
#     Convert PDF file to markdown using LangChain loaders.
#     Uses PyPDFLoader as primary, with UnstructuredPDFLoader as fallback.
#     """
#     try:
#         # Primary loader: PyPDFLoader
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#
#         # Convert documents to markdown format
#         markdown_content = ""
#         for i, doc in enumerate(documents):
#             page_num = i + 1
#             content = doc.page_content.strip()
#             if content:
#                 markdown_content += f"# Page {page_num}\n\n{content}\n\n"
#
#         return markdown_content.strip()
#
#     except Exception as e:
#         print(f"PyPDFLoader failed: {e}. Trying UnstructuredPDFLoader...")
#
#         try:
#             # Fallback loader: UnstructuredPDFLoader
#             loader = UnstructuredPDFLoader(file_path)
#             documents = loader.load()
#
#             # Convert documents to markdown format
#             markdown_content = ""
#             for i, doc in enumerate(documents):
#                 content = doc.page_content.strip()
#                 if content:
#                     # For unstructured loader, add section headers
#                     markdown_content += f"# Section {i + 1}\n\n{content}\n\n"
#             markdown_content = markdown_content.strip()
#             print(markdown_content)
#             return markdown_content
#
#         except Exception as fallback_error:
#             raise Exception(f"Both PDF loaders failed. PyPDFLoader: {e}, UnstructuredPDFLoader: {fallback_error}")
#
# def load_file_content(file_path: str) -> str:
#     """
#     Load file content. If PDF, convert to markdown first.
#     """
#     if file_path.lower().endswith('.pdf'):
#         print(f"Processing PDF file: {file_path}")
#     else:
#         # Read text files directly
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return f.read()
#
# def run_summarization_workflow(query: str, article: str, max_iterations: int = 4, requests_per_second: float | None = None):
#     """Run the iterative query-focused summarization workflow.
#
#     Args:
#         query: User query / focus.
#         article: Full article content (markdown or text).
#         max_iterations: Iteration cap.
#         requests_per_second: Rate limit applied per agent instance.
#     """
#     question_gen = QuestionGenerator(requests_per_second=requests_per_second)
#     summarizer = Summarizer(requests_per_second=requests_per_second)
#     qa_agent = QAAgentRunner(requests_per_second=requests_per_second)
#     judge_agent = Judge(requests_per_second=requests_per_second)
#
#     questions_output: QuestionsOutputType = question_gen.run(query=query, article=article)
#     # Keep full structured questions output; also retain the list for convenience
#     questions = questions_output.questions
#     acu_questions = getattr(questions_output, 'acu_questions', [])
#     current_summary = ""
#     sections_to_highlight = [] # For initial run, empty
#
#     # Initialize result structure for JSON output
#     workflow_result = {
#         "query": query,
#         "max_iterations": max_iterations,
#         "iterations": [],
#         "final_summary": "",
#         "total_iterations": 0,
#         "status": "",
#     # store initial generated questions
#     "questions": questions,
#     "acu_questions": acu_questions,
#     }
#
#     for iteration in range(max_iterations):
#         iteration_data = {
#             "iteration_number": iteration + 1,
#             "summary": "",
#             "qa_evaluations": [],  # full evaluation objects (dict-form)
#             "qa_pairs": [],         # simplified question-answer pairs for judge reference
#             "judge": None,
#             # counts to be filled post-judge
#             "correct_count_all": 0,
#             "correct_count_acu": 0,
#             "sections_to_highlight": [],
#             "sections_to_highlight_size": 0,
#         }
#         # 2. Summarizer
#         current_summary = summarizer.run(query=query, article=article, sections=sections_to_highlight)
#         iteration_data["summary"] = current_summary
#
#         # 3. QA Evaluations (includes answers + result + issue)
#         qa_evaluations_struct: QAAgentEvaluationsOutputType = qa_agent.run(questions_output=questions_output, summary=current_summary)
#         # Store full evaluations (model_dump for JSON serialization)
#         iteration_data["qa_evaluations"] = [ev.model_dump() for ev in qa_evaluations_struct.evaluations]
#         # Derive simple qa_pairs for Judge input
#         qa_pairs = [
#             {"question": ev.qa.question, "answer": ev.qa.answer} for ev in qa_evaluations_struct.evaluations
#         ]
#         iteration_data["qa_pairs"] = qa_pairs
#
#         # 4. Judge
#         judge_eval: JudgeEvaluationType = judge_agent.run(article=article, summary=current_summary, qa_pairs=qa_pairs)
#         iteration_data["judge"] = judge_eval.model_dump()
#         # capture judge-proposed sections for transparency in this iteration
#         iteration_data["sections_to_highlight"] = list(getattr(judge_eval, "sections_to_highlight", []))
#         iteration_data["sections_to_highlight_size"] = len(iteration_data["sections_to_highlight"]) if iteration_data["sections_to_highlight"] is not None else 0
#
#         # 5. Compute ACU metrics from judge evaluations
#         total_correct = sum(1 for ev in judge_eval.evaluations if ev.result is True)
#         acu_correct = 0
#         for ev in judge_eval.evaluations:
#             q_text = ev.qa.question if hasattr(ev, 'qa') else None
#             if isinstance(q_text, str) and q_text.strip().startswith("ACU."):
#                 if ev.result is True:
#                     acu_correct += 1
#         iteration_data["correct_count_all"] = total_correct
#         iteration_data["correct_count_acu"] = acu_correct
#         workflow_result["iterations"].append(iteration_data)
#
#         if judge_eval.judgment:
#             workflow_result["final_summary"] = current_summary
#             workflow_result["total_iterations"] = iteration + 1
#             workflow_result["status"] = "completed"
#             return workflow_result
#         else:
#             # Use judge-proposed sections directly
#             sections_to_highlight = list(getattr(judge_eval, "sections_to_highlight", []))
#
#     # Max iterations reached
#     workflow_result["final_summary"] = current_summary
#     workflow_result["total_iterations"] = max_iterations
#     workflow_result["status"] = "max_iterations_reached"
#     return workflow_result
#
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query-Focused Summarization Workflow")
    parser.add_argument('--file', type=str, required=True, help='Path to the article file (PDF or text)')
    parser.add_argument('--query', type=str, required=True, help='Query for summarization')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--json_path', type=str, required=False, help='If set, write JSON output directly to this file path')
    parser.add_argument('--limiter', type=float, required=False, help='If set, requests per second for each agent rate limiter.')

    args = parser.parse_args()


    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        exit(1)

    # Load article content from file (with PDF processing if needed)
    try:
        article_content = load_file_content(args.file)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

    result = run_summarization_workflow(
        query=args.query,
        article=article_content,
    max_iterations=args.max_iterations,
    requests_per_second=args.limiter if 'limiter' in args and args.limiter is not None else None,
    )

    # Always output JSON now
    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

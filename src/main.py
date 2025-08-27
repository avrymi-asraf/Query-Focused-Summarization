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

def process_pdf_to_markdown(file_path: str) -> str:
    """
    Convert PDF file to markdown using LangChain loaders.
    Uses PyPDFLoader as primary, with UnstructuredPDFLoader as fallback.
    """
    try:
        # Primary loader: PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Convert documents to markdown format
        markdown_content = ""
        for i, doc in enumerate(documents):
            page_num = i + 1
            content = doc.page_content.strip()
            if content:
                markdown_content += f"# Page {page_num}\n\n{content}\n\n"
        
        return markdown_content.strip()
        
    except Exception as e:
        print(f"PyPDFLoader failed: {e}. Trying UnstructuredPDFLoader...")
        
        try:
            # Fallback loader: UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            
            # Convert documents to markdown format
            markdown_content = ""
            for i, doc in enumerate(documents):
                content = doc.page_content.strip()
                if content:
                    # For unstructured loader, add section headers
                    markdown_content += f"# Section {i + 1}\n\n{content}\n\n"
            markdown_content = markdown_content.strip()
            print(markdown_content)
            return markdown_content
            
        except Exception as fallback_error:
            raise Exception(f"Both PDF loaders failed. PyPDFLoader: {e}, UnstructuredPDFLoader: {fallback_error}")

def load_file_content(file_path: str) -> str:
    """
    Load file content. If PDF, convert to markdown first.
    """
    if file_path.lower().endswith('.pdf'):
        print(f"Processing PDF file: {file_path}")
        return process_pdf_to_markdown(file_path)
    else:
        # Read text files directly
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

def run_summarization_workflow(query: str, article: str, max_iterations: int = 4, requests_per_second: float | None = None):
    question_gen = QuestionGenerator(requests_per_second=requests_per_second)
    summarizer = Summarizer(requests_per_second=requests_per_second)
    qa_agent = QAAgentRunner(requests_per_second=requests_per_second)
    judge_agent = Judge(requests_per_second=requests_per_second)

    questions_output: QuestionsOutputType = question_gen.run(query=query, article=article)
    questions = questions_output.questions
    acu_questions = getattr(questions_output, 'acu_questions', [])
    current_summary = ""
    sections_to_highlight: list[str] = []  # focus topics carried across iterations

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
        }

        # 1. Summarize (never sees raw questions)
        current_summary = summarizer.run(query=query, article=article, sections=sections_to_highlight)
        iteration_data["summary"] = current_summary

        # 2. QA over summary
        qa_evals_struct: QAAgentEvaluationsOutputType = qa_agent.run(
            questions_output=questions_output,
            summary=current_summary
        )
        iteration_data["qa_evaluations"] = [ev.model_dump() for ev in qa_evals_struct.evaluations]
        qa_pairs = [{"question": ev.qa.question, "answer": ev.qa.answer} for ev in qa_evals_struct.evaluations]
        iteration_data["qa_pairs"] = qa_pairs

        # 3. Judge vs full article
        judge_eval: JudgeEvaluationType = judge_agent.run(
            article=article,
            summary=current_summary,
            qa_pairs=qa_pairs
        )
        iteration_data["judge"] = judge_eval.model_dump()

        # 4. ACU miss seeding (only those explicitly flagged as missing atomic fact)
        missing_acu_concepts = [
            ev.qa.question.replace("ACU.", "").strip(" :?")
            for ev in judge_eval.evaluations
            if ev.qa.question.startswith("ACU.")
               and not ev.result
               and (ev.issue or "").lower().startswith("missing atomic fact")
        ]

        # 5. Metrics
        iteration_data["correct_count_all"] = sum(1 for ev in judge_eval.evaluations if ev.result)
        iteration_data["correct_count_acu"] = sum(
            1 for ev in judge_eval.evaluations
            if ev.qa.question.startswith("ACU.") and ev.result
        )

        workflow_result["iterations"].append(iteration_data)

        if judge_eval.judgment:
            workflow_result["final_summary"] = current_summary
            workflow_result["total_iterations"] = iteration + 1
            workflow_result["status"] = "completed"
            return workflow_result

        # 6. Build next focus topics (deduplicated, order preserved)
        next_sections: list[str] = []
        for concept in missing_acu_concepts:
            if concept and concept not in next_sections:
                next_sections.append(concept)

        for ev in judge_eval.evaluations:
            if not ev.result:
                topic = ev.issue or ev.qa.question
                if topic.startswith("ACU."):
                    topic = topic.replace("ACU.", "").strip(" :?")
                if topic and topic not in next_sections:
                    next_sections.append(topic)

        sections_to_highlight = next_sections  # carried into next loop

    workflow_result["final_summary"] = current_summary
    workflow_result["total_iterations"] = max_iterations
    workflow_result["status"] = "max_iterations_reached"
    return workflow_result

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

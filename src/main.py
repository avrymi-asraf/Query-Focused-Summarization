from Agents import QuestionGenerator, Summarizer, QAAgent, Judge
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

def run_summarization_workflow(query: str, article: str, max_iterations: int = 4, output_format: str = "print"):
    question_gen = QuestionGenerator()
    summarizer = Summarizer()
    qa_agent = QAAgent()
    judge_agent = Judge()

    questions = question_gen.run(query=query, article=article)
    current_summary = ""
    sections_to_highlight = [] # For initial run, empty
    
    # Initialize result structure for JSON output
    workflow_result = {
        "query": query,
        "max_iterations": max_iterations,
        "iterations": [],
        "final_summary": "",
        "total_iterations": 0,
        "status": ""
    }

    for iteration in range(max_iterations):
        iteration_data = {
            "iteration_number": iteration + 1,
            "summary": "",
            "qa_pairs": [],
            "needs_iteration": False,
            "missing_topics": []
        }
        
        if output_format == "print":
            print(f"\n--- Iteration {iteration + 1} ---")

        # 2. Summarizer
        current_summary = summarizer.run(article=article, sections=sections_to_highlight) #todo: maybe also send quary here?
        iteration_data["summary"] = current_summary
        
        if output_format == "print":
            print("Generated Summary (this iter):")
            # Format the summary for better readability
            formatted_summary = current_summary.replace("1. SUMMARY:", "\n1. SUMMARY:").replace("2. KEY HIGHLIGHTS:", "\n\n2. KEY HIGHLIGHTS:")
            # Add line breaks after bullet points and periods in highlights
            formatted_summary = formatted_summary.replace("* ", "\n* ").replace("• ", "\n• ")
            
            # Break up long paragraphs by adding line breaks after sentences
            import re
            # Add line breaks after sentences (period followed by space and capital letter)
            formatted_summary = re.sub(r'(\. )([A-Z])', r'\1\n\2', formatted_summary)
            # Also break after sentences ending with period at end of line
            formatted_summary = re.sub(r'(\.)( +)([A-Z])', r'\1\n\3', formatted_summary)
            
            print(formatted_summary)

        # 3. QA
        qa_pairs = qa_agent.run(questions=questions, summary=current_summary)
        iteration_data["qa_pairs"] = qa_pairs
        
        if output_format == "print":
            print("QA Pairs based on Summary (this iter):")
            for q, a in qa_pairs:
                print(f"Q: {q}\nA: {a}")

        # 4. Judge
        needs_iteration, missing_topics = judge_agent.run(
            article=article,
            summary=current_summary,
            qa_pairs=qa_pairs
        )
        
        iteration_data["needs_iteration"] = needs_iteration
        iteration_data["missing_topics"] = missing_topics
        workflow_result["iterations"].append(iteration_data)

        if not needs_iteration:
            workflow_result["final_summary"] = current_summary
            workflow_result["total_iterations"] = iteration + 1
            workflow_result["status"] = "completed"
            
            if output_format == "print":
                print("\nJudge satisfied! Summary is comprehensive.")
                return current_summary, iteration + 1
            else:
                return workflow_result
        else:
            if output_format == "print":
                print(f"\nJudge found missing topics. Needs another iteration. Missing topics: {missing_topics}")
            sections_to_highlight = missing_topics # Pass missing topics back for next summarization

    # Max iterations reached
    workflow_result["final_summary"] = current_summary
    workflow_result["total_iterations"] = max_iterations
    workflow_result["status"] = "max_iterations_reached"
    
    if output_format == "print":
        print(f"\nMax iterations ({max_iterations}) reached. Returning current summary.")
        return current_summary, max_iterations
    else:
        return workflow_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query-Focused Summarization Workflow")
    parser.add_argument('--file', type=str, required=True, help='Path to the article file (PDF or text)')
    parser.add_argument('--query', type=str, required=True, help='Query for summarization')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--output_format', type=str, choices=['print', 'json'], default='print', 
                       help='Output format: print for console output or json for structured data')
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
        output_format=args.output_format
    )
    
    if args.output_format == 'json':
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        final_summary, num_iters = result
        print("\nFinal Summary after workflow:")
        # Format the final summary for better readability
        formatted_final_summary = final_summary.replace("1. SUMMARY:", "\n1. SUMMARY:").replace("2. KEY HIGHLIGHTS:", "\n\n2. KEY HIGHLIGHTS:")
        # Add line breaks after bullet points and periods in highlights
        formatted_final_summary = formatted_final_summary.replace("* ", "\n* ").replace("• ", "\n• ")
        
        # Break up long paragraphs by adding line breaks after sentences
        import re
        # Add line breaks after sentences (period followed by space and capital letter)
        formatted_final_summary = re.sub(r'(\. )([A-Z])', r'\1\n\2', formatted_final_summary)
        # Also break after sentences ending with period at end of line
        formatted_final_summary = re.sub(r'(\.)( +)([A-Z])', r'\1\n\3', formatted_final_summary)
        
        print(formatted_final_summary)
        print(f"\nWorkflow completed in {num_iters} iterations.")

from Agents import QuestionGenerator, Summarizer, QAAgent, Judge
import argparse
import os
import json

def run_summarization_workflow(query: str, article: str, max_iterations: int = 5, output_format: str = "print"):
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
        current_summary = summarizer.run(article=article, sections=sections_to_highlight)
        iteration_data["summary"] = current_summary
        
        if output_format == "print":
            print("Generated Summary (this iter):")
            print(current_summary)

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
    parser.add_argument('--file', type=str, required=True, help='Path to the article file')
    parser.add_argument('--query', type=str, required=True, help='Query for summarization')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--output_format', type=str, choices=['print', 'json'], default='print', 
                       help='Output format: print for console output or json for structured data')
    args = parser.parse_args()


    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        exit(1)

    # Read article content from file
    with open(args.file, 'r', encoding='utf-8') as f:
        article_content = f.read()

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
        print(final_summary)
        print(f"Workflow completed in {num_iters} iterations.")

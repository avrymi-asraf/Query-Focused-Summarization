from Agents import QuestionGenerator, Summarizer, QAAgent, Judge
import argparse
import os

def run_summarization_workflow(query: str, article: str, max_iterations: int = 5):
    question_gen = QuestionGenerator()
    summarizer = Summarizer()
    qa_agent = QAAgent()
    judge_agent = Judge()

    questions = question_gen.run(query=query, article=article)
    current_summary = ""
    sections_to_highlight = [] # For initial run, empty

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # 2. Summarizer
        current_summary = summarizer.run(article=article, sections=sections_to_highlight)
        print("Generated Summary (this iter):")
        print(current_summary)

        # 3. QA
        qa_pairs = qa_agent.run(questions=questions, summary=current_summary)
        print("QA Pairs based on Summary (this iter):")
        for q, a in qa_pairs:
            print(f"Q: {q}\nA: {a}")

        # 4. Judge
        needs_iteration, missing_topics = judge_agent.run(
            article=article,
            summary=current_summary,
            qa_pairs=qa_pairs
        )

        if not needs_iteration:
            print("\nJudge satisfied! Summary is comprehensive.")
            return current_summary, iteration + 1
        else:
            print(f"\nJudge found missing topics. Needs another iteration. Missing topics: {missing_topics}")
            sections_to_highlight = missing_topics # Pass missing topics back for next summarization

    print(f"\nMax iterations ({max_iterations}) reached. Returning current summary.")
    return current_summary, max_iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query-Focused Summarization Workflow")
    parser.add_argument('--file', type=str, required=True, help='Path to the article file')
    parser.add_argument('--query', type=str, required=True, help='Query for summarization')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    args = parser.parse_args()


    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        exit(1)

    # Read article content from file
    with open(args.file, 'r', encoding='utf-8') as f:
        article_content = f.read()

    final_summary, num_iters = run_summarization_workflow(
        query=args.query,
        article=article_content,
        max_iterations=args.max_iterations
    )
    print("\nFinal Summary after workflow:")
    print(final_summary)
    print(f"Workflow completed in {num_iters} iterations.")

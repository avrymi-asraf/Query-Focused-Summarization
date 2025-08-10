import os
from typing import List, Tuple, Union, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Define the LLM instance to be reused
# Use the model with highest RPM/RPD for free tier
load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.233,
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=14,  # Controls the maximum burst size.
)

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", rate_limiter=rate_limiter)
_llm_summarizer = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_output_tokens=400, rate_limiter=rate_limiter)

# Helper function to extract text/content from various response types
def _extract_text(response: Union[str, Dict[str, Any]]) -> str:
    if isinstance(response, dict):
        # Prioritize 'content' for ChatMessage objects, then 'text'
        return response.get("content", response.get("text", ""))
    return str(response) # Ensure it's always a string

# --- Custom Output Parsers ---

class QuestionListParser(BaseOutputParser[List[str]]):
    """Parses a newline-separated string of questions into a list of strings."""
    def parse(self, text: str) -> List[str]:
        return [q.strip() for q in text.split("\n") if q.strip()]

class QAPairsParser(BaseOutputParser[List[Tuple[str, str]]]):
    """Parses a newline-separated string of 'question: answer' pairs."""
    def parse(self, text: str) -> List[Tuple[str, str]]:
        pairs = []
        for line in text.split("\n"):
            if ":" in line:
                q, a = line.split(":", 1)
                pairs.append((q.strip("- ").strip(), a.strip()))
        return pairs

class JudgeOutputParser(BaseOutputParser[Tuple[bool, List[str]]]):
    """Parses the Judge's response into (needs_iteration, missing_topics_list)."""
    def parse(self, text: str) -> Tuple[bool, List[str]]:
        reply = text.strip().upper()
        if reply == "OK":
            return False, []
        return True, [t.strip("- ") for t in text.split("\n") if t.strip()]

# --- Agents using LCEL ---

class QuestionGenerator:
    def __init__(self, llm=None):
        self.llm = llm or _llm # Use the global _llm if not provided
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "Given the user query:\n"
             "{query}\n"
             "and the article:\n"
             "{article}\n"
             "Generate exactly 10 diagnostic questions that help assess understanding of the article in relation to the query.\n\n"
             "Guidelines for questions:\n"
             "- Include a mix of factual, analytical, and inferential questions\n"
             "- Ensure questions cover different aspects/sections of the article\n"
             "- Make questions specific and directly answerable from the article content\n"
             "- Vary complexity from straightforward recall to deeper analysis\n"
             "- All questions must be relevant to both the article content and user query\n\n"
             "Format: Output ONLY the 10 questions, one per line, without numbering or any additional text."
             )
        ])
        self.chain = self.prompt | self.llm | StrOutputParser() | QuestionListParser()

    def run(self, query: str, article: str) -> List[str]:
        return self.chain.invoke({"query": query, "article": article})

class Summarizer:
    def __init__(self, llm=None):
        self.llm = llm or _llm_summarizer
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "Summarize the following article in a concise, informative manner.\n\n"
             "Article:\n{article}\n\n"
             "Key points to include (if provided, otherwise identify them yourself):\n{sections}\n\n"
             "Format your response as follows:\n"
             "1. SUMMARY: A cohesive 200-250 word overview capturing ALL main ideas, key points and conclusions\n"
             "2. KEY HIGHLIGHTS: 3-5 concise statements (50-100 words total) highlighting the most important facts, data points, or claims, ensuring ALL critical information is covered\n\n"
             "Focus on accuracy and factual information from article only.\n"
             "For lengthy articles, prioritize the most significant content, and key points.\n"
             "Provide ONLY the formatted summary and highlights without additional commentary.\n"
            )
        ])
        # sections will be passed as a newline-separated string or empty
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, article: str, sections: List[str]) -> str:
        # LCEL automatically handles passing inputs as dict
        return self.chain.invoke({"article": article, "sections": "\n".join(sections)})

class QAAgent:
    def __init__(self, llm=None):
        self.llm = llm or _llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "Based STRICTLY on this summary (do not use outside knowledge):\n"
             "{summary}\n\n"
             "Answer each question below. Follow these rules:\n"
             "- If the summary contains a direct answer, provide it concisely\n"
             "- If the summary has partial information, provide what's available\n"
             "- If the summary has no relevant information, respond EXACTLY with 'Not enough information in summary'\n"
             "- Do not speculate or infer beyond what's explicitly stated\n\n"
             "Format each response as 'Question: Answer' pairs (one pair per line, with the colon separator).\n\n"
             "Questions:\n{questions}"
            )
        ])
        self.chain = (
            {"questions": RunnableLambda(lambda x: "\n".join(f"- {q}" for q in x["questions"])),
             "summary": RunnablePassthrough()} # Pass summary through
            | self.prompt
            | self.llm
            | StrOutputParser()
            | QAPairsParser()
        )

    def run(self, questions: List[str], summary: str) -> List[Tuple[str, str]]:
        # Pass a dictionary for multiple inputs
        return self.chain.invoke({"questions": questions, "summary": summary})


class Judge:
    def __init__(self, llm=None):
        self.llm = llm or _llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "# Evaluation Task\n\n"
             "Article:\n{article}\n\n"
             "Summary:\n{summary}\n\n"
             "QA pairs (from summary):\n{qa_pairs}\n\n"
             "## Instructions\n"
             "You are a critical judge evaluating the quality and completeness of the summary and answers. Analyze:\n"
             "1. Are all key facts, concepts, and major arguments from the article covered in the summary?\n"
             "2. Are the QA answers accurate according to the original article (not just the summary)?\n"
             "3. Are there any important numerical data, dates, names, or specific details missing?\n\n"
             "If BOTH the summary is comprehensive AND the QA pairs accurately reflect the article, respond with EXACTLY 'OK'.\n"
             "Otherwise, list each missing or incorrectly addressed topic on a new line with a hyphen, focusing on substance rather than style."
            )
        ])
        self.chain = (
            {"qa_pairs": RunnableLambda(lambda x: "\n".join(f"{q}: {a}" for q, a in x["qa_pairs"])),
             "article": RunnablePassthrough(),
             "summary": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
            | JudgeOutputParser()
        )

    def run(self, article: str, summary: str, qa_pairs: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
        # Pass a dictionary for multiple inputs
        return self.chain.invoke({"article": article, "summary": summary, "qa_pairs": qa_pairs})


if __name__ == "__main__":
    # Dummy data for testing
    pass
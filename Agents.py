import os
from typing import List, Tuple, Union, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
# Define the LLM instance to be reused
# Use the model with highest RPM/RPD for free tier
load_dotenv()
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

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
             "Given the query:\n"
             "{query}\n"
             "and the article:\n"
             "{article}\n"
             "Generate diagnostic questions, one per line. Do not include any introductory or concluding remarks."
            )
        ])
        self.chain = self.prompt | self.llm | StrOutputParser() | QuestionListParser()

    def run(self, query: str, article: str) -> List[str]:
        return self.chain.invoke({"query": query, "article": article})

class Summarizer:
    def __init__(self, llm=None):
        self.llm = llm or _llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "Summarize the following article and highlight key sections. "
             "If no sections are provided, start fresh.\n\n"
             "Article:\n{article}\n\n"
             "Sections to consider (if any, otherwise ignore):\n{sections}\n\n"
             "Provide only the summary and highlighted sections, no other text."
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
             "Based only on this summary:\n"
             "{summary}\n\n"
             "Answer each question below. If the summary does not contain enough information "
             "to answer a question, state 'Not enough information in summary'.\n"
             "Format your response as 'Question: Answer' pairs, one pair per line.\n\n"
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
             "Article:\n{article}\n\n"
             "Summary:\n{summary}\n\n"
             "QA pairs (from summary):\n{qa_pairs}\n\n"
             "Based on the full article, list any missing important topics or key facts that are NOT covered in the summary. "
             "If the summary is comprehensive and covers all important aspects, respond with ONLY ‘OK’. "
             "Otherwise, list each missing topic/fact on a new line, prefixed with a hyphen."
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
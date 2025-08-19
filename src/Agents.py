import os
from typing import List, Union, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Environment & shared LLM setup
# ---------------------------------------------------------------------------
load_dotenv()
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.233,
    check_every_n_seconds=0.1,
    max_bucket_size=14,
)
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
_llm_summarizer = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class QuestionsOutput(BaseModel):
    questions: List[str] = Field(..., description="Exactly 5 diagnostic questions relevant to the query and article.")


class QAAgent:
    class QAPair(BaseModel):
        question: str = Field(..., description="Original question (verbatim)")
        answer: str = Field(..., description="Answer derived ONLY from the summary or fallback text if insufficient info")

    class QAPairsOutput(BaseModel):
        pairs: List["QAAgent.QAPair"]


class QuestionEvaluation(BaseModel):
    qa: QAAgent.QAPair
    result: bool
    issue: Optional[str] = None


class JudgeEvaluation(BaseModel):
    evaluations: List[QuestionEvaluation]
    judgment: bool

class QAAgentEvaluationsOutput(BaseModel):
    """Structured output for QAAgentRunner: evaluations over questions based only on the summary."""
    evaluations: List[QuestionEvaluation]



# ---------------------------------------------------------------------------
# Question Generator
# ---------------------------------------------------------------------------
class QuestionGenerator:
    def __init__(self, llm=None):
        base_llm = llm or _llm
        self.llm = base_llm.with_structured_output(QuestionsOutput)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                "You are an expert question formulator with skills in critical analysis and comprehension testing.\n\n"
                "User Query:\n{query}\n\n"
                "Article Content:\n{article}\n\n"
                "Task: Generate exactly 5 diverse, diagnostic questions that evaluate understanding of the article in relation to the user query.\n\n"
                "Guidelines:\n"
                "- Mix factual, analytical, and inferential perspectives\n"
                "- Cover different important sections/aspects\n"
                "- Each question must be answerable directly from the article\n"
                "- No duplicate focus; vary depth and angle\n"
                "- Must stay tightly relevant to the query AND the article\n\n"
                "Return ONLY valid JSON for the schema with key 'questions'."
            )
        ])
        self.chain = self.prompt | self.llm

    def run(self, query: str, article: str) -> QuestionsOutput:
        return self.chain.invoke({"query": query, "article": article})


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------
class Summarizer:
    def __init__(self, llm=None):
        self.llm = llm or _llm_summarizer
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                "You are an expert summarizer tasked with creating a summary of an article from a specific user's perspective.\n\n"
                "User's Query/Perspective:\n{query}\n\n"
                "Given the article:\n{article}\n\n"
                "In this iteration, specifically focus on these topics (if provided):\n{sections}\n\n"
                "Format your response as follows:\n"
                "1. SUMMARY: A cohesive 200-250 word overview that directly addresses the user's query.\n"
                "2. KEY HIGHLIGHTS: 3-5 concise statements highlighting the most important facts relevant to the query.\n\n"
                "Focus ONLY on information relevant to the user's query. Provide ONLY the formatted summary and highlights."
            )
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query: str, article: str, sections: List[str]) -> str:
        return self.chain.invoke({
            "query": query,
            "article": article,
            "sections": "\n".join(sections) if sections else "(none)"
        })


# ---------------------------------------------------------------------------
# QA Agent (simplified structured output)
# ---------------------------------------------------------------------------

class QAAgentRunner:
    def __init__(self, llm=None):
        base_llm = llm or _llm
        # Ask the model directly for evaluations (question+answer+result+issue)
        self.llm = base_llm.with_structured_output(QAAgentEvaluationsOutput)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                "You are a careful answer extractor and adequacy assessor.\n\n"
                "Summary (ONLY source of truth):\n{summary}\n\n"
                "For EACH question do ALL of the following USING ONLY the summary:\n"
                "1. Provide an answer derived strictly from the summary.\n"
                "2. Determine if the summary provides enough information for a reliable answer.\n"
                "3. If there is NO relevant info, answer EXACTLY: Not enough information in summary\n"
                "4. Set result=false when the answer is the fallback OR when information is clearly partial/insufficient.\n"
                "5. When result=false give a short issue explaining what is missing (or 'Not enough information').\n\n"
                "Rules:\n"
                "- Never speculate beyond the summary.\n"
                "- If partial data exists, answer ONLY what is present and set result=false with an issue like 'partial information'.\n"
                "- If sufficient and direct, set result=true and leave issue null.\n\n"
                "Return ONLY valid JSON with key 'evaluations'. Each element object must match schema:\n"
                "{{ 'qa': {{ 'question': <original>, 'answer': <answer> }}, 'result': <bool>, 'issue': <string|null> }}\n\n"
                "Questions:\n{questions}"
            )
        ])
        self.chain = (
            {
                "questions": RunnableLambda(lambda x: "\n".join(f"- {q}" for q in x["questions_list"])),
                "summary": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

    def run(self, questions_output: QuestionsOutput, summary: str) -> QAAgentEvaluationsOutput:
        return self.chain.invoke({
            "questions_list": questions_output.questions,
            "summary": summary
        })


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
class Judge:
    def __init__(self, llm=None):
        self.llm = (llm or _llm).with_structured_output(JudgeEvaluation)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                "You are a critical evaluator focused on factual accuracy, completeness, and specificity.\n\n"
                "Article (reference):\n{article}\n\n"
                "Summary (to evaluate):\n{summary}\n\n"
                "QA pairs (from summary):\n{qa_pairs}\n\n"
                "Evaluate each question-answer ONLY using the article. For each pair: set result=true if the answer is accurate, complete, and specific; else result=false with a concise explanation in 'issue'.\n"
                "Overall 'judgment' is true only if ALL answers pass AND the summary has no major omissions or factual errors.\n\n"
                "Return ONLY structured JSON per schema."
            )
        ])
        self.chain = (
            {
                "qa_pairs": RunnableLambda(lambda x: "\n".join(f"{p['question']}: {p['answer']}" for p in x["qa_pairs"])),
                "article": RunnablePassthrough(),
                "summary": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

    def run(self, article: str, summary: str, qa_pairs: List[Dict[str, str]]) -> JudgeEvaluation:
        return self.chain.invoke({
            "article": article,
            "summary": summary,
            "qa_pairs": qa_pairs
        })


__all__ = [
    "QuestionsOutput",
    "QuestionGenerator",
    "Summarizer",
    "QAAgent",
    "QAAgentRunner",
    "QAAgentEvaluationsOutput",
    "QuestionEvaluation",
    "JudgeEvaluation",
    "Judge",
]


if __name__ == "__main__":
    pass
"""Agent & model definitions.

Each agent now creates its own LLM instance internally using a simple
`InMemoryRateLimiter` built from the value passed in via the constructor.
No shared globals and no helper factory functions (kept intentionally simple).
"""
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field

load_dotenv()

# Data Models ----------------------------------------------------------------
class QuestionsOutputType(BaseModel):
    questions: List[str] = Field(
        ..., description="Exactly 5 diagnostic questions relevant to the query and article."
    )

class QAPairType(BaseModel):
    question: str = Field(..., description="Original question (verbatim)")
    answer: str = Field(..., description="Answer derived ONLY from the summary or fallback text if insufficient info")

class QAPairsOutputType(BaseModel):
    pairs: List[QAPairType]

class QuestionEvaluationType(BaseModel):
    qa: QAPairType
    result: bool
    issue: Optional[str] = None

class JudgeEvaluationType(BaseModel):
    evaluations: List[QuestionEvaluationType]
    judgment: bool

class QAAgentEvaluationsOutputType(BaseModel):
    evaluations: List[QuestionEvaluationType]

class QAAgent:  # Placeholder
    pass

# Question Generator ---------------------------------------------------------
class QuestionGenerator:
    """
    Produces:
      - 5 diagnostic questions.
      - Up to 3 ACU questions (prefixed 'ACU.') that target ONLY easy, high‑salience,
        explicitly stated atomic facts (numbers, dates, named entities) directly relevant
        to the user query. Excludes anything requiring inference or aggregation.
    """
    def __init__(self, requests_per_second: Optional[float]):
        if requests_per_second is not None:
            limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=14,
            )
            base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", rate_limiter=limiter)
        else:
            base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        self.llm = base_llm.with_structured_output(QuestionsOutputType)
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "You are an expert question formulator.\n"
             "User Query:\n{query}\n\n"
             "Article:\n{article}\n\n"
             "Produce JSON with keys 'questions' no more then 7 questions.\n\n"
             "QUESTIONS:\n"
             "- Diverse (coverage, perspective, depth).\n"
             "- Directly answerable from the article.\n"
             "- Strongly tied to BOTH query and content.\n"
             "- No overlap or trivial rephrases.\n\n"
             "Return ONLY valid JSON for schema."
             "REMEMBER: The questions should be directly tied to the query and content. and answerable from the article."
             "Article:\n{article}\n\n"
             "User Query:\n{query}\n\n")
        ])
        self.chain = self.prompt | self.llm

    def run(self, query: str, article: str) -> QuestionsOutputType:
        return self.chain.invoke({"query": query, "article": article})


# Summarizer -----------------------------------------------------------------
class Summarizer:
    def __init__(self, requests_per_second: Optional[float]):
        if requests_per_second is not None:
            limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=14,
            )
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", rate_limiter=limiter)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "You are an expert summarizer tasked with creating a summary of an article from a specific user's perspective.\n\n"
             "User's Query/Perspective:\n{query}\n\n"
             "Given the article:\n{article}\n\n"
             "In this iteration, specifically focus on these topics (if provided):\n{sections}\n\n"
             "Format your response as follows:\n"
             "1. SUMMARY: A cohesive 200-250 word overview that directly addresses the user's query.\n"
             "2. KEY HIGHLIGHTS: 3-5 concise statements highlighting the most important facts relevant to the query.\n\n"
             "Focus ONLY on information relevant to the user's query. Provide ONLY the formatted summary and highlights.")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query: str, article: str, sections: List[str]) -> str:
        return self.chain.invoke({
            "query": query,
            "article": article,
            "sections": "\n".join(sections) if sections else "(none)"
        })

# QA Agent Runner ------------------------------------------------------------
class QAAgentRunner:
    """
    Answers both regular and ACU questions ONLY from the summary.
    ACU success depends on the fact being naturally embedded in the summary text.
    """
    def __init__(self, requests_per_second: Optional[float]):
        if requests_per_second is not None:
            limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=14,
            )
            base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", rate_limiter=limiter)
        else:
            base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.llm = base_llm.with_structured_output(QAAgentEvaluationsOutputType)
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "You answer strictly from the SUMMARY text below.\n\n"
             "SUMMARY:\n{summary}\n\n"
             "For each question:\n"
             "- Use ONLY information present in the summary.\n"
             "- If question starts with 'ACU.' extract the precise minimal span (number/date/name). If absent verbatim or clearly equivalent, answer exactly: Not enough information in summary\n"
             "- For non‑ACU: provide best exact answer; if missing, same fallback.\n"
             "- result=true only if answer is fully supported & specific. Otherwise result=false and issue describes ('missing', 'partial', or 'Not enough information').\n"
             "- Never invent.\n\n"
             "Return JSON with key 'evaluations'.\n"
             "Questions:\n{questions}")
        ])
        self.chain = (
            {
                "questions": RunnableLambda(lambda x: "\n".join(f"- {q}" for q in x["questions_list"])),
                "summary": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

    def run(self, questions_output: QuestionsOutputType, summary: str) -> QAAgentEvaluationsOutputType:
        combined = list(questions_output.questions) + list(questions_output.acu_questions)
        return self.chain.invoke({"questions_list": combined, "summary": summary})

# Judge ----------------------------------------------------------------------
class Judge:
    """
    Verifies QA against the full article. For ACU: checks that the atomic fact
    exists in the article and (if present in article) also appears in the summary answer.
    Guides iteration by flagging missing atomic facts -> feed into next 'sections'.
    """
    def __init__(self, requests_per_second: Optional[float]):
        if requests_per_second is not None:
            limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=14,
            )
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", rate_limiter=limiter)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.llm = llm.with_structured_output(JudgeEvaluationType)
        self.prompt = ChatPromptTemplate.from_messages([
            ("human",
             "You evaluate QA factuality vs the ARTICLE (ground truth) and assess SUMMARY completeness indirectly.\n\n"
             "ARTICLE:\n{article}\n\n"
             "SUMMARY:\n{summary}\n\n"
             "QA PAIRS:\n{qa_pairs}\n\n"
             "For each pair:\n"
             "- If question begins 'ACU.':\n"
             "  * If article states the atomic fact and answer matches (allow trivial formatting), result=true.\n"
             "  * If article states it but answer missing/inexact/fallback -> result=false, issue='missing atomic fact'.\n"
             "  * If article does NOT contain it and answer claims a fact -> result=false, issue='unsupported'.\n"
             "  * If article lacks it and answer correctly used fallback -> result=true.\n"
             "- Non‑ACU: result=true only if accurate & specific; else false with brief issue.\n"
             "- Keep issues concise.\n"
             "judgment=true only if ALL results true AND no major obvious omissions (e.g., repeatedly central atomic facts absent from summary).\n\n"
             "Return JSON per schema.")
        ])
        self.chain = ({
            "qa_pairs": RunnableLambda(lambda x: "\n".join(f"{p['question']}: {p['answer']}" for p in x["qa_pairs"])),
            "article": RunnablePassthrough(),
            "summary": RunnablePassthrough(),
        } | self.prompt | self.llm)

    def run(self, article: str, summary: str, qa_pairs: List[Dict[str, str]]) -> JudgeEvaluationType:
        return self.chain.invoke({"article": article, "summary": summary, "qa_pairs": qa_pairs})

__all__ = [
    "QuestionsOutputType",
    "QuestionGenerator",
    "Summarizer",
    "QAAgent",
    "QAAgentRunner",
    "QAAgentEvaluationsOutputType",
    "QuestionEvaluationType",
    "JudgeEvaluationType",
    "Judge",
    "QAPairType",
    "QAPairsOutputType",
]

if __name__ == "__main__":  # pragma: no cover
    pass
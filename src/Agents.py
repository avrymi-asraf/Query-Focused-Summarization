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
    acu_questions: List[str] = Field(
        default_factory=list,
        description=(
            "0-5 ACU questions ONLY when appropriate. Each MUST start with 'ACU.' and ask for "
            "a single short fact (number, date, named entity, or atomic fact) that is strictly "
            "relevant to the user's query and is obvious and unambiguous in the article. If no "
            "such ACUs exist, this list MUST be empty."
        ),
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
    sections_to_highlight: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete sections/topics WITH an actionable angle for the next iteration;"
            " format each as 'Heading — focus' (or 'Topic — instruction'); concise;"
            " do not paste QA issue texts."
        ),
    )

class QAAgentEvaluationsOutputType(BaseModel):
    evaluations: List[QuestionEvaluationType]

class QAAgent:  # Placeholder
    pass

# Question Generator ---------------------------------------------------------
class QuestionGenerator:
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
            (
                "human",
                "You are an expert question formulator with skills in critical analysis and comprehension testing.\n\n"
                "User Query:\n{query}\n\n"
                "Article Content:\n{article}\n\n"
                "Task: Produce BOTH of the following:\n"
                "- Exactly 5 diverse diagnostic questions in 'questions' that evaluate understanding of the article in relation to the user query.\n"
                "- 0 to 5 ACU questions in 'acu_questions' ONLY IF they meet all criteria below; otherwise return an empty list. ACU questions MUST begin with 'ACU.' and request a single short, unambiguous fact (number, date, named entity, or atomic fact) that directly connects to the query.\n\n"
                "Guidelines for 'questions':\n"
                "- Mix factual, analytical, and inferential perspectives\n"
                "- Cover different important sections/aspects\n"
                "- Each question must be answerable directly from the article\n"
                "- No duplicate focus; vary depth and angle\n"
                "- Must stay tightly relevant to the query AND the article\n\n"
                "Guidelines for 'acu_questions':\n"
                "- Each MUST start with 'ACU.' (e.g., 'ACU. What year was X founded?')\n"
                "- STRICT FILTER: Include ACUs only when the fact is (a) strictly relevant to the user's query, (b) explicitly present in the article, and (c) obvious and unambiguous. If any doubt exists, include none.\n"
                "- The answer must be a single short value (number, date, named entity, or atomic fact); avoid multi-part or compound questions.\n"
                "- Do not infer or aggregate; the fact must be explicitly stated and directly tied to the query.\n"
                "- If no appropriate ACUs exist, output an empty list for 'acu_questions'.\n\n"
                "REMEMBER: Ask simple questions that are directly related to the query and that are answered in the article!."
                "Return ONLY valid JSON adhering to the schema with keys 'questions' and 'acu_questions'."
            )
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
             "Schema example (literals, not placeholders): {{ 'qa': {{ 'question': <original>, 'answer': <answer> }}, 'result': <bool>, 'issue': <string|null> }}\n\n"
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
        # Combine regular and ACU questions (ACUs are already prefixed with 'ACU.' for later identification)
        combined = list(questions_output.questions) + list(questions_output.acu_questions)
        return self.chain.invoke({"questions_list": combined, "summary": summary})

# Judge ----------------------------------------------------------------------
class Judge:
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
             "You are a critical evaluator focused on factual accuracy, completeness, and specificity.\n\n"
             "Article (reference):\n{article}\n\n"
             "Summary (to evaluate):\n{summary}\n\n"
             "QA pairs (from summary):\n{qa_pairs}\n\n"
             "Evaluate each question-answer ONLY using the article. For each pair: set result=true if the answer is accurate, complete, and specific; else result=false with a concise explanation in 'issue'.\n"
             "Overall 'judgment' is true only if ALL answers pass AND the summary has no major omissions or factual errors.\n\n"
             "Then, propose 3-7 'sections_to_highlight' for the next summarization iteration.\n"
             "Each item must be a concrete section/topic PLUS a short angle for what to extract, formatted as 'Heading — focus'.\n"
             "Examples: 'Methodology — key assumptions and limitations'; 'Results — core metrics vs baseline'.\n"
             "Anchor to exact article headings if present; otherwise create a short explicit topic label.\n"
             "Cover gaps revealed by failed/partial QA WITHOUT copying issue text. Be specific, actionable, and non-redundant.\n"
             "Keep each item brief (~5–12 words).\n\n"
             "Return ONLY structured JSON with keys: evaluations, judgment, sections_to_highlight.")
        ])
        self.chain = ({
            "qa_pairs": RunnableLambda(lambda x: "\n".join(f"{p['question']}: {p['answer']}" for p in x["qa_pairs"])),
            "article": RunnablePassthrough(),
            "summary": RunnablePassthrough(),
        } | self.prompt | self.llm)

    def run(self, article: str, summary: str, qa_pairs: List[Dict[str, str]]) -> JudgeEvaluationType:
        return self.chain.invoke({
            "article": article,
            "summary": summary,
            "qa_pairs": qa_pairs
        })

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
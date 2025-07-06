from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def _extract_text(response):
    if isinstance(response, dict):
        return response.get("text", response.get("content", ""))
    return response

class QuestionGenerator:
    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
        prompt = PromptTemplate.from_template(
            "Given the query:\n"
            "{query}\n"
            "and the article:\n"
            "{article}\n"
            "Generate diagnostic questions, one per line."
        )
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def run(self, query: str, article: str) -> list[str]:
        resp = self.chain.invoke({"query": query, "article": article})
        text = _extract_text(resp)
        return [q.strip() for q in text.split("\n") if q.strip()]

class Summarizer:
    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
        prompt = PromptTemplate.from_template(
            "Summarize the following article and highlight key sections:\n\n"
            "{article}\n\n"
            "If no sections provided, start fresh."
        )
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def run(self, article: str, sections: list[str]) -> str:
        resp = self.chain.invoke({"article": article, "sections": "\n".join(sections)})
        return _extract_text(resp)

class QAAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
        prompt = PromptTemplate.from_template(
            "Based only on this summary:\n"
            "{summary}\n\n"
            "Answer each question below:\n"
            "{questions}\n\n"
            "Return as question: answer pairs."
        )
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def run(self, questions: list[str], summary: str) -> list[tuple[str, str]]:
        resp = self.chain.invoke({
            "questions": "\n".join(f"- {q}" for q in questions),
            "summary": summary
        })
        raw = _extract_text(resp)
        pairs = []
        for line in raw.split("\n"):
            if ":" in line:
                q, a = line.split(":", 1)
                pairs.append((q.strip("- "), a.strip()))
        return pairs

class Judge:
    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
        prompt = PromptTemplate.from_template(
            "Article:\n{article}\n\n"
            "Summary:\n{summary}\n\n"
            "QA pairs:\n{qa_pairs}\n\n"
            "List any missing topics not covered in the summary. "
            "If none, respond with ‘OK’."
        )
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def run(self, article: str, summary: str, qa_pairs: list[tuple[str, str]]):
        qa_text = "\n".join(f"{q}: {a}" for q, a in qa_pairs)
        resp = self.chain.invoke({
            "article": article,
            "summary": summary,
            "qa_pairs": qa_text
        })
        reply = _extract_text(resp).strip()
        if reply.upper() == "OK":
            return False, []
        return True, [t.strip("- ") for t in reply.split("\n") if t.strip()]

# pip install -U langchain-google-genai

from dotenv import load_dotenv
import langchain as lch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from Agents import QuestionGenerator, Summarizer, QAAgent, Judge

def run_pipeline(query: str, article: str, max_iter: int = 3) -> str:
    qg = QuestionGenerator()
    summarizer = Summarizer()
    qa_agent = QAAgent()
    judge = Judge()
    sections: list[str] = []

    for _ in range(max_iter):
        questions = qg.run(query, article)
        summary = summarizer.run(article, sections)
        qa_pairs = qa_agent.run(questions, summary)
        continue_loop, missing_topics = judge.run(article, summary, qa_pairs)
        if not continue_loop:
            return summary
        sections.extend(missing_topics)

    return summary

if __name__ == '__main__':
    load_dotenv()

    user_query = "Explain large language models in one sentence."
    # article_text = "..."  # load or fetch your article here
    article_text = "As teachers, it is important to know exactly where your students are at in their learning, while at the same time, identifying children who are at risk of falling behind. For students right at the beginning of their schooling, it is important this happens as early as possible, so that teachers can address each student’s individual learning needs. The Australian Early Development Census (AECD) provides some rich data on this and every 3 years, they publish fresh findings that show exactly how students are doing in 5 key domains. The domains are proven to positively impact mental health, wellbeing and educational outcomes in later life. In this article, we look at the results to come from the 2024 study, what this means for teachers, and uncover opportunities for further action. Research shows that when children thrive in their early years, they have a strong foundation for lifelong learning, health, development and wellbeing."

    final_summary = run_pipeline(user_query, article_text)

    print(final_summary)

    #
    # # # deal with getting input from user
    #
    # # Initialize the Chat Model (for conversational use cases)
    # # You can specify different Gemini models here, e.g., "gemini-2.5-flash", "gemini-1.5-pro", etc.
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17") # RPM is high
    #
    # # Example text invocation
    # response = llm.invoke("Explain large language models in one sentence.")
    # print(response.content)

    # Example for multimodal input (if using a vision-capable model like gemini-pro-vision or relevant flash models)
    # message = HumanMessage(
    #     content=[
    #         {"type": "text", "text": "What's in this image?"},
    #         {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    #     ]
    # )
    # response = llm.invoke([message])
    # print(response.content)







# from langchain import ChatOpenAI
# from dotenv import load_dotenv
# import os
#
#
# load_dotenv()
# llm = ChatOpenAI(
#     model="pgt-4-turbo",
#     api_key=os.getenv(API_OAI),
#     temperature=os.getenv("TEMPERATURE")
#     )
#
# response = llm.invoke("is a cucumber more long than green?")
# print(response)

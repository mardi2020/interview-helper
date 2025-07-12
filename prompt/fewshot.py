from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate


def get_feedback_prompt():
    examples = [
        {
            "answer": "FastAPI에서 async/await을 사용해 LLM 호출을 병렬 처리했습니다.",
            "feedback": "좋은 접근입니다. 어떤 방식으로 병렬화했는지 더 구체적으로 설명하면 더 좋겠습니다. 예: asyncio.gather 사용 여부 등"
        },
        {
            "answer": "RAG 챗봇에서 LangChain을 썼어요.",
            "feedback": "LangChain 사용 언급은 좋지만, 어떤 모듈을 사용했는지, chunk 전략이나 retriever 방식에 대한 구체적인 설명이 필요합니다."
        },
    ]

    example_prompt = PromptTemplate(
        input_variables=["answer", "feedback"],
        template="지원자: {answer}\n피드백: {feedback}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="당신은 기술 면접관입니다. 지원자의 답변을 보고 피드백을 제공합니다.",
        suffix="지원자: {user_answer}\n피드백:",
        input_variables=["user_answer"]
    )

    return prompt

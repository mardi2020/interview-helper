from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate
from langchain.schema.runnable import RunnableLambda
from config.parameters import get_llm
from langchain.globals import set_debug

llm = get_llm()



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

    fewshot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        # prefix="당신은 기술 면접관입니다. 지원자의 답변을 보고 피드백을 제공합니다.",
        suffix="지원자: {user_answer}\n피드백:",
        input_variables=["user_answer"],
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
                      당신은 기술 면접관입니다.
                      다음 지원자와의 대화를 바탕으로 지원자의 답변을 보고 피트백을 제공합니다.
                      지원자와의 대화 내역:
                      """),
        MessagesPlaceholder(variable_name="chat_history"),
        SystemMessage(content="\n"),
        SystemMessagePromptTemplate.from_template("{examples_output}"),
    ])
    
    chain = (
        RunnableLambda(lambda x: {"examples_output": fewshot_prompt.format(user_answer=x["user_answer"]), "chat_history": x["chat_history"]})
        | prompt
    )

    return chain


# def get_feedback(user_input: str) -> str:
#     prompt = get_feedback_prompt()
#     formatted_prompt = prompt.format(user_answer=user_input)
#     response = llm.invoke(formatted_prompt)
#     feedback_text = response.content.strip() if hasattr(response, "content") else str(response)
#     return feedback_text


def get_feedback(user_input: str, history: BaseChatMessageHistory) -> str:
    prompt = get_feedback_prompt()
    # formatted_prompt = prompt.format(user_answer=user_input)
    # response = llm.invoke(formatted_prompt)
    
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,  # 세션 기록을 가져오는 함수
        input_messages_key="user_answer",
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    # print(chain_with_history.get_prompts(config={"configurable": {"session_id": "test"}}))
    set_debug(True)
    response = chain_with_history.invoke({"user_answer": user_input}, config={"configurable": {"session_id": "test"}})
    feedback_text = response.content.strip() if hasattr(response, "content") else str(response)
    return feedback_text

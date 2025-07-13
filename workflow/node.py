"""
Interview agents module: defines functions for asking questions, giving feedback,
and summarizing interview sessions using LLM and vector search.
"""

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

from config.parameters import get_llm
from rag.vector_store import load_faiss

from .state import InterviewState



def ask_agent(state: InterviewState) -> InterviewState:
    """
    Generate a technical interview question based on resume and conversation.

    Args:
        state (InterviewState): Current interview state including messages and keywords.

    Returns:
        InterviewState: Updated interview state with a new question added.
    """

    system_prompt = """
    당신은 전문적인 기술 면접관입니다.
    뛰어난 인재를 선발하기 위해 이력서 기반으로 날카로운 질문을 합니다.
    """

    messages = [SystemMessage(content=system_prompt)]

    for message in state["messages"]:
        if message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        else:
            messages.append(
                HumanMessage(content=f"{message['role']}: {message['content']}")
            )
    try:
        db = load_faiss()
        retriever = db.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.get_relevant_documents(" ".join(state["tech_keywords"]))
        context = "\n\n".join(doc.page_content for doc in context_docs)
        context_text = f"""다음은 지원자의 이력 기반 정보입니다:\n{context}"""
    except FileNotFoundError:
        context_text = "지원자의 문서 기반 정보가 없습니다. 기술 키워드만 참고하세요."


    prompt = f"""
    아래 이력 정보를 참고해,  
    다음 기술({", ".join(state["tech_keywords"])}) 관련 정보를 알아보세요.
    그 후 이력 정보와 연결하여 관련 면접 질문을 설명없이 한글로, 한 문장으로 생성하세요.
    과거에 질문했던 질문은 하지 말아주세요.
        
    이력 정보:  
    {context_text}
    """

    messages.append(HumanMessage(content=prompt))

    tools = load_tools(tool_names=["arxiv", "wikipedia"], llm=get_llm())
    agent = initialize_agent(
        tools=tools, llm=get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #    verbose=True
    )

    response = agent.invoke(messages)
    new_state = state.copy()

    new_state["messages"].append({"role": "interviewer", "content": response['output']})
    return new_state


def feedback_agent(state: InterviewState) -> InterviewState:
    """
    Provide feedback on candidate's answers using few-shot prompting.

    Args:
        state (InterviewState): Current interview state including user input and message history.

    Returns:
        InterviewState: Updated interview state with feedback added.
    """

    system_prompt = """
    당신은 전문적인 기술 면접관이자 강사입니다.
    취업준비생들에게 조언을 해주기 위해 질문/답변들을 토대로 조언을 해줍니다.
    """
    messages = [SystemMessage(content=system_prompt)]
    # state에서 메시지 가져오기
    for message in state["messages"]:
        if message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        else:
            messages.append(
                HumanMessage(content=f"{message['role']}: {message['content']}")
            )

    examples = [
        {
            "answer": "FastAPI에서 async/await을 사용해 LLM 호출을 병렬 처리했습니다.",
            "feedback": "좋은 접근입니다. 어떤 방식으로 병렬화했는지 더 구체적으로 설명하면 더 좋겠습니다. 예: asyncio.gather 사용 여부 등"
        },
        {
            "answer": "RAG 챗봇에서 LangChain을 썼어요.",
            "feedback": (
                "LangChain 사용 언급은 좋지만, 어떤 모듈을 사용했는지, "
                 "chunk 전략이나 retriever 방식에 대한 구체적인 설명이 필요합니다."
                )
        },
    ]

    example_prompt = PromptTemplate(
        input_variables=["answer", "feedback"],
        template="지원자: {answer}\n피드백: {feedback}"
    )

    fewshot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="지원자의 답변을 보고 피드백을 제공합니다.",
        suffix="지원자: {user_answer}\n피드백:",
        input_variables=["user_answer"],
    )
    messages.append(HumanMessage(content=fewshot_prompt.invoke(
        {"user_answer": state["user_input"]}).to_string())
        )

    response = get_llm().invoke(messages)

    new_state = state.copy()
    new_state["messages"].append({"role": "applicant", "content": state["user_input"]})
    new_state["messages"].append({"role": "feedback", "content": response.content})
    return new_state


def summary_agent(state: InterviewState) -> InterviewState:
    """
    Summarize technical weaknesses and improvements from the session.

    Args:
        state (InterviewState): Current interview state including message history.

    Returns:
        InterviewState: Updated interview state with summary added.
    """

    system_prompt = """
    당신은 전문적인 기술 면접관입니다.
    """

    messages = [SystemMessage(content=system_prompt)]

    for message in state["messages"]:
        if message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        else:
            messages.append(
                HumanMessage(content=f"{message['role']}: {message['content']}")
            )

    prompt = """
        위의 내용은 사용자 질문/답변/피드백의 기록입니다. 이 데이터를 바탕으로 사용자의 기술적 약점과 개선점을 요약해주세요.

        요약:
    """
    messages.append(
        HumanMessage(content=prompt)
    )
    response = get_llm().invoke(messages)
    new_state = state.copy()

    new_state["messages"].append({"role": "summarier", "content": response.content})
    return new_state

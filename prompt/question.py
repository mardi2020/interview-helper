from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, BaseMessage
from rag.vector_store import load_faiss
from config.parameters import get_llm

llm = get_llm()


def get_question_prompt(context: str, tech_keywords: list[str]) -> str:
    return f"""
당신은 기술 면접관입니다. 아래 이력 기반 정보와 대화 맥락을 참고해,  
다음 기술({', '.join(tech_keywords)}) 관련 면접 질문을 하나만,  
질문만, 설명 없이, 한 문장으로 생성하세요.
    
이력 정보:  
{context}

이전 대화: 참고용
"""


def generate_question_with_rag(tech_keywords: list[str], history: list[BaseMessage]) -> str:
    try:
        db = load_faiss()
        retriever = db.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.get_relevant_documents(" ".join(tech_keywords))
        context = "\n\n".join(doc.page_content for doc in context_docs)
        context_text = f"""다음은 지원자의 이력 기반 정보입니다:\n{context}"""
    except FileNotFoundError:
        context_text = "지원자의 문서 기반 정보가 없습니다. 기술 키워드만 참고하세요."

    system_prompt = get_question_prompt(context_text, tech_keywords)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        SystemMessage(content="위 정보를 바탕으로 다음 질문을 작성하세요.")
    ])

    messages = prompt.format_messages(history=history)

    return llm.invoke(messages).content.strip()

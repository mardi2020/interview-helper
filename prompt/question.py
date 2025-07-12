from langchain.prompts import PromptTemplate
from config.parameters import get_llm
from rag.vector_store import load_faiss

llm = get_llm()

question_prompt = PromptTemplate(
    input_variables=["tech_list"],
    template="""
당신은 신입/주니어 백엔드 개발자를 위한 기술 면접관입니다.

다음 기술 스택을 기반으로 하나의 면접 질문을 만들어주세요:
{tech_list}

조건:
- 너무 쉽지 않고, 개념을 확인할 수 있는 질문
- 한 문장으로 명확하게
- 오픈형 질문 (예/아니오가 아닌 설명을 요구)

질문:
"""
)


def generate_question(tech_keywords: list[str]) -> str:
    tech_list_str = "\n- " + "\n- ".join(tech_keywords)
    prompt = question_prompt.format(tech_list=tech_list_str)
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_question_with_rag(tech_keywords: list[str]) -> str:
    db = load_faiss()
    context = ""

    if db is not None:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.get_relevant_documents(" ".join(tech_keywords))
        context = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""당신은 기술 면접관입니다.

다음은 지원자의 이력 기반 정보입니다:
{context if context else '[문서 없음]'}

이 내용을 바탕으로 다음 기술({', '.join(tech_keywords)}) 관련 면접 질문을 하나 생성해주세요.

조건:
- 설명형 질문
- 한 문장
- 너무 쉽지 않도록

질문:"""

    return llm.invoke(prompt).content.strip()

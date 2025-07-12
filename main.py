from langchain_core.messages import HumanMessage, AIMessage
from prompt.fewshot import get_feedback_prompt
from prompt.question import generate_question_with_rag
from config.parameters import get_llm
from rag.loader import load_and_split_file
from rag.vector_store import save_to_faiss

import streamlit as st

llm = get_llm()


def init_session():
    st.session_state.stage = "select_topic"
    st.session_state.uploaded_file_name = None
    st.session_state.selected_topics = []
    st.session_state.messages = []
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.feedbacks = []


def render_ui(page_title="나의 면접관"):
    st.set_page_config(page_title=page_title)
    st.title(page_title)

    # 세션 초기화
    if "messages" not in st.session_state:
        init_session()

    # 문서 업로드 (RAG)
    st.sidebar.header("📄 문서 업로드 (RAG)")
    uploaded_file = st.sidebar.file_uploader(
        "이력서, 포트폴리오 등 질문받고 싶은 문서를 업로드하세요. (pdf, txt, md)",
        type=["pdf", "txt", "md"]
    )
    if uploaded_file:
        if uploaded_file.name != st.session_state.get("uploaded_file_name"):
            with st.spinner("문서를 처리 중입니다..."):
                chunks = load_and_split_file(uploaded_file)
                save_to_faiss(chunks)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.sidebar.success(f"{len(chunks)}개의 문서 청크가 벡터 DB에 저장되었습니다.")
                st.rerun()

    # 대화 초기화
    if st.button("🧹 대화 초기화"):
        st.session_state.clear()
        init_session()
        st.rerun()

    # 기술 선택 단계
    if st.session_state.stage == "select_topic":
        tech_input = st.text_input("연습하고 싶은 기술을 입력하세요 (예: Java, SpringBoot, Database)")
        if st.button("시작하기") and tech_input.strip():
            st.session_state.selected_topics = [t.strip() for t in tech_input.split(",") if t.strip()]
            st.session_state.stage = "ask"
            st.rerun()

    # 질문 생성 단계
    elif st.session_state.stage == "ask":
        question = generate_question_with_rag(st.session_state.selected_topics)
        st.session_state.questions.append(question)
        st.session_state.messages.append(AIMessage(content=question))
        st.session_state.stage = "wait_answer"
        st.rerun()

    # 기존 대화 렌더링
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # 사용자 답변 입력창: wait_answer 단계에서만 출력
    user_input = None
    if st.session_state.stage == "wait_answer":
        user_input = st.chat_input("질문에 답변해보세요!", key="user_input")

    # 답변 처리 및 피드백
    if st.session_state.stage == "wait_answer" and user_input:
        st.session_state.answers.append(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        feedback_prompt = get_feedback_prompt().format(user_answer=user_input)
        feedback = llm.invoke(feedback_prompt).content
        st.session_state.feedbacks.append(feedback)
        st.session_state.messages.append(AIMessage(content=feedback))

        st.session_state.stage = "confirm_next"
        st.rerun()

    # 다음 질문 여부
    elif st.session_state.stage == "confirm_next":
        st.markdown("다음 질문을 이어서 진행할까요?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ 네, 다음 질문"):
                st.session_state.stage = "ask"
                st.rerun()
        with col2:
            if st.button("🛑 그만할게요"):
                st.session_state.stage = "summary"
                st.rerun()

    # 요약 출력
    elif st.session_state.stage == "summary":
        with st.spinner("당신의 답변을 바탕으로 부족한 부분을 요약 중입니다..."):
            qa_text = "\n\n".join(
                f"Q: {q}\nA: {a}\nFeedback: {f}"
                for q, a, f in zip(
                    st.session_state.questions,
                    st.session_state.answers,
                    st.session_state.feedbacks
                )
            )
            summary_prompt = f"""
당신은 기술 면접관입니다. 아래는 사용자의 면접 답변과 그에 대한 피드백입니다.
이 데이터를 바탕으로 사용자의 전반적인 기술적 약점과 개선점을 요약해주세요.

답변 및 피드백:
{qa_text}

요약:
"""
            summary = llm.invoke(summary_prompt).content
            st.markdown("### 📋 면접 피드백 요약")
            st.markdown(summary)


if __name__ == "__main__":
    render_ui()

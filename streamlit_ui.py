"""
Streamlit UI module for the interview chatbot application.

Contains functions to initialize session state, render UI components
based on the current workflow stage, and handle user interactions.

The UI flow covers:
- Document upload for retrieval-augmented generation (RAG)
- Technical topic selection
- Interview question generation and display
- User answer input and AI feedback generation
- Session summary display

All UI rendering functions accept:
- st: The Streamlit module, passed from the main application for flexibility.
- graph: The interview workflow StateGraph instance.
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from rag.loader import load_and_split_file
from rag.vector_store import save_to_faiss



def init_session(st):
    """
    Initialize all Streamlit session state variables for a fresh chatbot session.

    Args:
        st: The Streamlit module for session state management.
    """
    st.session_state.stage = "select_topic"
    st.session_state.uploaded_file_name = None
    st.session_state.selected_topics = []
    st.session_state.messages = []
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.feedbacks = []

    st.session_state.memory = ChatMessageHistory()
    st.session_state.graph_state = {
        "messages": [],
        "tech_keywords": "",
        "is_summary": False,
        "user_input": ""
    }
    st.session_state.graph_config = {"configurable": {"thread_id": "unique_session"}}


def render_document_upload(st):
    """
    Render the sidebar UI to upload documents (pdf, txt, md)
    and save their chunks to the FAISS vector store.

    Args:
        st: The Streamlit module for UI rendering.
    """
    st.sidebar.header("📄 문서 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "이력서, 포트폴리오 등 질문받고 싶은 문서를 업로드하세요. (pdf, txt, md)",
        type=["pdf", "txt", "md"]
    )
    if uploaded_file and uploaded_file.name != st.session_state.get("uploaded_file_name"):
        with st.spinner("문서를 처리 중입니다..."):
            chunks = load_and_split_file(uploaded_file)
            save_to_faiss(chunks)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.sidebar.success(f"{len(chunks)}개의 문서 청크가 벡터 DB에 저장되었습니다.")
        st.rerun()


def render_select_topic(st):
    """
    Render UI for the user to input and select technical topics to practice.

    Args:
        st: The Streamlit module for UI rendering.
    """
    tech_input = st.text_input("연습하고 싶은 기술을 입력하세요 (예: Java, SpringBoot, MySQL 등)")
    if st.button("시작하기") and tech_input.strip():
        st.session_state.selected_topics = [t.strip() for t in tech_input.split(",") if t.strip()]
        st.session_state.stage = "ask"
        st.session_state.graph_state["tech_keywords"] = st.session_state.selected_topics
        st.rerun()


def render_ask(st, graph):
    """
    Generate an interview question by invoking the graph at the 'ask' step,
    display it, and update session state accordingly.

    Args:
        st: The Streamlit module for UI rendering.
        graph: The workflow StateGraph instance managing the interview process.
    """
    graph.update_state(values=st.session_state.graph_state, config=st.session_state.graph_config)
    question_prompt = graph.invoke(None,
                                   interrupt_after="ask",
                                   config=st.session_state.graph_config)
    st.session_state.graph_state = question_prompt
    question = question_prompt["messages"][-1]["content"]
    st.session_state.questions.append(question)
    st.session_state.messages.append(AIMessage(content=question))
    st.session_state.stage = "wait_answer"
    st.rerun()


def render_messages(st):
    """
    Render the chat messages (user and assistant) in the Streamlit chat UI.

    Args:
        st: The Streamlit module for UI rendering.
    """
    if st.session_state.messages:
        for message in st.session_state.messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)


def render_wait_answer(st, graph):
    """
    Display an input box for the user to answer the current question,
    send the answer to the graph for feedback generation, and update the session.

    Args:
        st: The Streamlit module for UI rendering.
        graph: The workflow StateGraph instance managing the interview process.
    """
    user_input = st.chat_input("질문에 답변해보세요.", key="user_input")
    if user_input:
        st.session_state.answers.append(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.graph_state["user_input"] = user_input
        graph.update_state(values=st.session_state.graph_state,
                           config=st.session_state.graph_config)
        response = graph.invoke(None, interrupt_after="feedback",
                                config=st.session_state.graph_config)
        st.session_state.graph_state = response
        feedback = response['messages'][-1]['content']
        st.session_state.feedbacks.append(feedback)
        st.session_state.messages.append(AIMessage(content=feedback))
        st.session_state.stage = "confirm_next"
        st.rerun()


def render_confirm_next(st):
    """
    Ask the user whether to continue with the next question or finish the session.

    Args:
        st: The Streamlit module for UI rendering.
    """
    st.markdown("다음 질문을 이어서 진행할까요?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ 네, 다음 질문"):
            st.session_state.stage = "ask"
            st.rerun()
    with col2:
        if st.button("🛑 그만할게요"):
            st.session_state.stage = "summary"
            st.session_state.graph_state["is_summary"] = True
            st.rerun()


def render_summary(st, graph):
    """
    Generate and display a summary feedback based on the entire interview session.

    Args:
        st: The Streamlit module for UI rendering.
        graph: The workflow StateGraph instance managing the interview process.
    """
    with st.spinner("당신의 답변을 바탕으로 부족한 부분을 요약 중입니다..."):
        graph.update_state(values=st.session_state.graph_state,
                           config=st.session_state.graph_config)
        response = graph.invoke(None, config=st.session_state.graph_config)
        summary = response['messages'][-1]['content']
        st.markdown("### 📋 면접 피드백 요약")
        st.markdown(summary)


def render_ui(st, graph, page_title="나의 면접관"):
    """
    Main function to render the entire Streamlit UI for the interview chatbot,
    routing UI rendering to the appropriate stage-specific functions.

    Args:
        st: The Streamlit module for UI rendering.
        graph: The workflow StateGraph instance managing the interview process.
        page_title (str): Title displayed on the Streamlit app page.
    """
    st.set_page_config(page_title=page_title)
    st.title(page_title)

    if "messages" not in st.session_state:
        init_session(st)

    render_document_upload(st)

    stage = st.session_state.stage

    if stage == "select_topic":
        render_select_topic(st)
    elif stage == "ask":
        render_ask(st, graph)
    elif stage == "wait_answer":
        render_messages(st)
        render_wait_answer(st, graph)
    elif stage == "confirm_next":
        render_messages(st)
        render_confirm_next(st)
    elif stage == "summary":
        render_summary(st, graph)

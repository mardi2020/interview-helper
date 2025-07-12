from langchain_core.prompts import PromptTemplate


def get_summary_prompt():
    return PromptTemplate.from_template("""
당신은 빅테크의 Back-end 기술 면접관입니다. 아래는 사용자 질문/답변/피드백의 기록입니다. 이 데이터를 바탕으로 사용자의 기술적 약점과 개선점을 요약해주세요.

{qa_history}

요약:
""")

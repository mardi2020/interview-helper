from typing import Dict, List, TypedDict
from operator import add
class CurrentStep:
    ASK = "ASK"
    FEEDBACK = "FEEDBACK"
    SUMMARY = "SUMMARY"
    USER_INPUT="USER_INPUT"

class InterviewState(TypedDict):
    
    messages: List[Dict]
    tech_keywords: List[str]
    is_summary: bool
    user_input: str
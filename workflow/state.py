"""
Module defining constants and state types for managing the interview session workflow.

Includes:
- CurrentStep: Class with string constants representing interview stages.
- InterviewState: TypedDict describing the shape of the interview state data.
"""

from typing import Dict, List, TypedDict


class CurrentStep:
    """
    Constants representing different stages of the interview process.
    """

    ASK = "ASK"
    FEEDBACK = "FEEDBACK"
    SUMMARY = "SUMMARY"
    USER_INPUT="USER_INPUT"


class InterviewState(TypedDict):
    """
    TypedDict describing the structure of the interview state.

    Attributes:
        messages (List[Dict]): List of message dictionaries containing roles and contents.
        tech_keywords (List[str]): List of technical keywords relevant to the interview.
        is_summary (bool): Flag indicating if the current step is the summary phase.
        user_input (str): The latest input from the user.
    """

    messages: List[Dict]
    tech_keywords: List[str]
    is_summary: bool
    user_input: str

"""
Module to create and compile the interview workflow graph
using StateGraph with defined agents and conditional transitions.
"""

from langgraph.graph import StateGraph, END, START

from workflow.node import ask_agent, feedback_agent, summary_agent
from workflow.state import InterviewState



def create_graph() -> StateGraph:
    """
    Create a StateGraph representing the interview process workflow.

    Nodes:
        - "ask": Handles asking questions via ask_agent.
        - "feedback": Handles user feedback via feedback_agent.
        - "summary": Handles summary generation via summary_agent.

    Edges:
        - START -> "ask"
        - "ask" -> "feedback"
        - Conditional edge from "feedback" to "summary" or back to "ask" based on 'is_summary' flag.
        - "summary" -> END

    Returns:
        Compiled StateGraph ready for execution.
    """
    workflow = StateGraph(InterviewState)

    workflow.add_node("ask", ask_agent)
    workflow.add_node("feedback", feedback_agent)
    workflow.add_node("summary", summary_agent)

    workflow.add_edge(START, "ask")
    workflow.add_edge("ask", "feedback")

    workflow.add_conditional_edges(
        "feedback",
        lambda s: (
            "summary" if s["is_summary"] else "ask"
        ),
        ["ask", "summary"],
    )

    workflow.add_edge("summary", END)
    return workflow.compile()

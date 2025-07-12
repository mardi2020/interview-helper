from langgraph.graph import StateGraph, END, START

from workflow.node import ask_agent, feedback_agent, summary_agent
from workflow.state import InterviewState

def create_graph() -> StateGraph:
    workflow = StateGraph(InterviewState)
    
    workflow.add_node("ask", ask_agent)
    workflow.add_node("feedback", feedback_agent)
    workflow.add_node("summary", summary_agent)
   
    workflow.add_edge(START, "ask")
    workflow.add_edge("ask", "feedback")
    # workflow.add_edge("feedback", "ask")
    
    workflow.add_conditional_edges(
        "feedback",
        lambda s: (
            "summary" if s["is_summary"] else "ask"
        ),
        ["ask", "summary"],
    )
    
    workflow.add_edge("summary", END)
    return workflow.compile()
    
    
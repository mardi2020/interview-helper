"""
Main entry point for the interview helper Streamlit app.

Creates the interview workflow graph and passes it along with the Streamlit
module to the UI renderer.
"""
import streamlit as st

from streamlit_ui import render_ui
from workflow.graph import get_graph



if __name__ == "__main__":
    render_ui(st, get_graph())

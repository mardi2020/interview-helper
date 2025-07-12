"""
Main entry point for the interview helper Streamlit app.

Creates the interview workflow graph and passes it along with the Streamlit
module to the UI renderer.
"""
from streamlit_ui import render_ui

import streamlit as st

from workflow.graph import create_graph


graph = create_graph()


if __name__ == "__main__":
    render_ui(st, graph)

import streamlit as st
from core.session_manager import initialize_session
from core.database import get_session_schema_info
from core.notebook import render_notebook_interface
from utils.helpers import setup_logging


def main():
    """Main Streamlit application entry point"""
    st.set_page_config(page_title="Data Analysis Notebook", layout="wide")

    # Setup logging
    setup_logging()

    # App header
    st.title("📓 Data Analysis Notebook")
    st.subheader("Session-based data analysis with natural language!")

    # Initialize session (database, cleanup, etc.)
    if not initialize_session():
        st.error("Failed to initialize session. Please refresh the page.")
        st.stop()

    # Load session schema info if not already loaded
    if "session_schema_info" not in st.session_state:
        st.session_state.session_schema_info = get_session_schema_info(
            st.session_state.session_db_conn
        )

    # Render the main notebook interface (includes sidebar + cells)
    render_notebook_interface()


if __name__ == "__main__":
    main()
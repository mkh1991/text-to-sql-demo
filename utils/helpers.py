import streamlit as st
from loguru import logger
import sys


def setup_logging():
    """Configure logging for the application"""
    # Remove default logger
    logger.remove()

    # Add console logger with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file logger for debugging (optional)
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def format_dataframe_preview(df, max_rows=3, max_cols=5):
    """Format DataFrame for preview display"""
    if df is None or df.empty:
        return "No data available"

    preview_df = df.head(max_rows)
    if len(df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
        cols_hidden = len(df.columns) - max_cols
        note = f" (+ {cols_hidden} more columns)"
    else:
        note = ""

    return preview_df, note


def truncate_text(text, max_length=100):
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_cell_status_emoji(status):
    """Get emoji for cell status"""
    status_map = {"empty": "⚪", "running": "🔄", "complete": "✅", "error": "❌"}
    return status_map.get(status, "⚪")


def format_row_count(count):
    """Format row count for display"""
    if count >= 1000000:
        return f"{count / 1000000:.1f}M"
    elif count >= 1000:
        return f"{count / 1000:.1f}K"
    else:
        return str(count)


def validate_session_state():
    """Validate that required session state exists"""
    required_keys = [
        "session_id",
        "session_db_conn",
        "session_db_path",
        "notebook_cells",
        "cell_counter",
    ]

    missing_keys = [key for key in required_keys if key not in st.session_state]

    if missing_keys:
        logger.warning(f"Missing session state keys: {missing_keys}")
        return False

    return True


def safe_execute(func, default_return=None, error_message="An error occurred"):
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        st.error(f"{error_message}: {str(e)}")
        return default_return

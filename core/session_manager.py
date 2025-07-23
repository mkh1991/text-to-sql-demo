import streamlit as st
import uuid
import sqlite3
import pandas as pd
import requests
from pathlib import Path
import time
from loguru import logger
import os


PARSE_DATES = ["Order Date", "Ship Date"]


def get_or_create_session_id():
    """Get or create a unique session ID"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[
                                      :8]  # Short UUID for readability
    return st.session_state.session_id


def setup_session_database(session_id):
    """Setup session-specific database with original data"""
    try:
        # Create sessions directory if it doesn't exist
        sessions_dir = Path("sessions")
        sessions_dir.mkdir(exist_ok=True)

        # Session database path
        session_db_path = sessions_dir / f"session_{session_id}.db"

        # If session DB already exists, just connect to it
        if session_db_path.exists():
            conn = sqlite3.connect(str(session_db_path), check_same_thread=False)
            logger.info(f"Connected to existing session DB: {session_db_path}")
            return conn, session_db_path

        # Create new session database
        logger.info(f"Creating new session DB: {session_db_path}")

        # File path for the CSV
        csv_file_path = "superstore.csv"
        url = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"

        # Check if CSV file already exists locally
        if os.path.exists(csv_file_path):
            st.write("Loading data from local CSV file...")
            df = pd.read_csv(csv_file_path, parse_dates=PARSE_DATES)
        else:
            st.write("Downloading CSV file...")
            try:
                # Download the CSV file
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Save to local file
                with open(csv_file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                # Read from the saved file
                df = pd.read_csv(csv_file_path, parse_dates=PARSE_DATES)
                st.write("CSV file downloaded and saved successfully!")

            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading file: {e}")
                return None
            except Exception as e:
                st.error(f"Error saving or reading file: {e}")
                return None

        st.write(df.head())

        # # Clean column names
        # df.columns = [
        #     col.replace(" ", "_").replace("-", "_").lower() for col in df.columns
        # ]

        # Create session SQLite database
        conn = sqlite3.connect(str(session_db_path), check_same_thread=False)

        # Load original data into session database
        df.to_sql("superstore", conn, if_exists="replace", index=False)

        logger.info(f"Session database created successfully: {session_db_path}")
        return conn, session_db_path

    except Exception as e:
        st.error(f"Error setting up session database: {str(e)}")
        return None, None


def cleanup_old_sessions(max_age_seconds=60*1):
    """Clean up old session files"""
    try:
        sessions_dir = Path("sessions")
        if not sessions_dir.exists():
            return

        current_time = time.time()

        cleaned_count = 0
        for db_file in sessions_dir.glob("session_*.db"):
            file_age = current_time - db_file.stat().st_mtime
            if file_age > max_age_seconds:
                db_file.unlink()
                cleaned_count += 1
                logger.info(f"Cleaned up old session: {db_file}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old session(s)")

    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")


def initialize_session():
    """Initialize session with database and cleanup"""
    # Get or create session ID
    session_id = get_or_create_session_id()

    # Cleanup old sessions on startup
    cleanup_old_sessions()

    # Initialize session database
    if "session_db_conn" not in st.session_state:
        with st.spinner(f"Setting up session workspace ({session_id})..."):
            conn, db_path = setup_session_database(session_id)
            if conn:
                st.session_state.session_db_conn = conn
                st.session_state.session_db_path = db_path
                st.success(f"Session workspace ready! Session ID: {session_id}")
            else:
                st.error("Failed to setup session workspace")
                return False

    return True


def reset_session():
    """Start a new session"""
    # Clear session state to force new session
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def clear_session_data():
    """Clear all session data and delete session file"""
    # Close connection and delete session file
    if hasattr(st.session_state, 'session_db_conn'):
        st.session_state.session_db_conn.close()
    if hasattr(st.session_state, 'session_db_path'):
        Path(st.session_state.session_db_path).unlink(missing_ok=True)
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import requests
from io import StringIO

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Initialize the Gemini GenerativeModel client
# Use 'gemini-1.5-flash-latest' for the latest Flash model

client = instructor.from_provider(model="google/gemini-2.0-flash")

import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


class SQLQuery(BaseModel):
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Plain English explanation of what the query does")
    confidence: float = Field(description="Confidence score between 0 and 1")
    tables_used: List[str] = Field(description="List of tables used in the query")

def setup_database():
    """Download and setup the superstore database"""
    try:
        # Download superstore dataset
        url = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))

        # Create SQLite database
        conn = sqlite3.connect('superstore.db', check_same_thread=False)

        # Clean column names (replace spaces with underscores)
        df.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df.columns]

        print(df.head())

        # Load data into SQLite
        df.to_sql('superstore', conn, if_exists='replace', index=False)

        return conn, df.columns.tolist()

    except Exception as e:
        st.error(f"Error setting up database: {str(e)}")
        return None, []

def get_schema_info(conn):
    """Get database schema information"""
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(superstore)")
        columns = cursor.fetchall()

        schema_info = "Database Schema:\n"
        schema_info += "Table: superstore\n"
        schema_info += "Columns:\n"

        for col in columns:
            schema_info += f"  - {col[1]} ({col[2]})\n"

        # Add sample data
        cursor.execute("SELECT * FROM superstore LIMIT 3")
        sample_data = cursor.fetchall()

        schema_info += "\nSample Data (first 3 rows):\n"
        for row in sample_data[:1]:  # Just show first row to keep prompt concise
            schema_info += f"  {dict(zip([col[1] for col in columns], row))}\n"

        return schema_info

    except Exception as e:
        return f"Error getting schema: {str(e)}"

def generate_sql_query(question: str, schema_info: str) -> SQLQuery:
    """Generate SQL query using Gemini with Instructor"""

    system_prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries.

{schema_info}

Rules:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Use proper SQL syntax for SQLite
3. Be careful with column names - use exact names from schema
4. For date operations, use SQLite date functions
5. Always include a confidence score based on query complexity and ambiguity

Example queries:
- "What are the top 5 products by sales?" -> SELECT product_name, SUM(sales) as total_sales FROM superstore GROUP BY product_name ORDER BY total_sales DESC LIMIT 5
- "Show sales by category" -> SELECT category, SUM(sales) as total_sales FROM superstore GROUP BY category ORDER BY total_sales DESC

Human question: {question}"""

    try:
        response = client.messages.create(
            messages=[{"role": "user", "content": system_prompt}],
            response_model=SQLQuery,
        )
        return response
    except Exception as e:
        print(e)
        return SQLQuery(
            sql="",
            explanation=f"Error generating query: {str(e)}",
            confidence=0.0,
            tables_used=[]
        )

def execute_query(conn, sql: str) -> Dict[str, Any]:
    """Execute SQL query and return results"""
    try:
        # Basic safety check
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith('SELECT'):
            return {"success": False, "error": "Only SELECT queries are allowed", "data": None}

        df = pd.read_sql_query(sql, conn)
        return {
            "success": True,
            "data": df,
            "error": None,
            "row_count": len(df)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None,
            "row_count": 0
        }

def main():
    st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")

    st.title("🔍 Text-to-SQL Agent")
    st.markdown("Ask questions about the Superstore dataset in natural language!")

    # Initialize database
    if 'db_conn' not in st.session_state:
        with st.spinner("Setting up database..."):
            conn, columns = setup_database()
            if conn:
                st.session_state.db_conn = conn
                st.session_state.schema_info = get_schema_info(conn)
                st.success("Database loaded successfully!")
            else:
                st.error("Failed to load database")
                return

    # Sidebar with schema info and examples
    with st.sidebar:
        st.header("📊 Database Info")
        with st.expander("View Schema", expanded=False):
            st.text(st.session_state.schema_info)

        st.header("💡 Example Questions")
        example_questions = [
            "What are the top 10 products by sales?",
            "Show monthly sales trends for 2017",
            "Which customers have the highest profit?",
            "What's the profit margin by category?",
            "Show sales performance by region",
            "What are the most discounted products?"
        ]

        for example in example_questions:
            if st.button(example, key=f"example_{example}"):
                st.session_state.user_question = example

    # Main query interface
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.text_input(
            "Ask a question about the data:",
            value=st.session_state.get('user_question', ''),
            placeholder="e.g., What are the top selling products in California?"
        )

    with col2:
        st.write("")  # Spacing
        query_button = st.button("Generate Query", type="primary")

    if query_button and user_question:
        with st.spinner("Generating SQL query..."):
            # Generate SQL
            sql_response = generate_sql_query(user_question, st.session_state.schema_info)

            if sql_response.sql:
                # Display generated query
                st.subheader("Generated SQL Query")
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.code(sql_response.sql, language="sql")

                with col2:
                    st.metric("Confidence", f"{sql_response.confidence:.2f}")

                st.write(f"**Explanation:** {sql_response.explanation}")

                # Execute query
                with st.spinner("Executing query..."):
                    result = execute_query(st.session_state.db_conn, sql_response.sql)

                    if result["success"]:
                        st.subheader("Query Results")
                        st.write(f"Found {result['row_count']} rows")

                        if result["row_count"] > 0:
                            st.dataframe(result["data"], use_container_width=True)

                            # Simple visualization for numeric data
                            numeric_cols = result["data"].select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0 and len(result["data"]) > 1:
                                st.subheader("Quick Visualization")

                                # Try to create a simple chart
                                if len(result["data"]) <= 20:  # Only for small result sets
                                    try:
                                        if len(result["data"].columns) == 2:
                                            st.bar_chart(result["data"].set_index(result["data"].columns[0]))
                                        elif len(numeric_cols) == 1:
                                            st.bar_chart(result["data"][numeric_cols[0]])
                                    except:
                                        pass  # Skip visualization if it fails
                        else:
                            st.info("Query executed successfully but returned no results.")
                    else:
                        st.error(f"Query execution failed: {result['error']}")
            else:
                st.error("Failed to generate SQL query")

if __name__ == "__main__":
    main()
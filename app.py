import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import sqlite3
import pandas as pd
import google.generativeai as genai
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import requests
from io import StringIO
from loguru import logger

# Configure Gemini
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except StreamlitSecretNotFoundError:
    api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=api_key)
# Initialize the Gemini GenerativeModel client
# Use 'gemini-2.5-flash-latest' for the latest Flash model
client = instructor.from_provider(model="google/gemini-2.5-flash")


class SQLQuery(BaseModel):
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Plain English explanation of what the query does"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    tables_used: List[str] = Field(description="List of tables used in the query")


class QueryonData(BaseModel):
    status: bool = Field(
        "Whether the query is relevant to the dataset and is answerable"
    )
    answer: str = Field(
        description="Detailed answer to the query, using multiple "
        "bullet points if necessary"
    )
    explanation: str = Field(
        description="Explanation for the answer or for why the query is not "
        "answerable for this dataset"
    )


def setup_database():
    """Download and setup the superstore database"""
    try:
        # Download superstore dataset
        url = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))

        # Create SQLite database
        conn = sqlite3.connect("superstore.db", check_same_thread=False)

        # Clean column names (replace spaces with underscores)
        df.columns = [
            col.replace(" ", "_").replace("-", "_").lower() for col in df.columns
        ]

        logger.info(df.head())

        # Load data into SQLite
        df.to_sql("superstore", conn, if_exists="replace", index=False)

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
        cursor.execute("SELECT * FROM superstore LIMIT 1")
        sample_data = cursor.fetchall()

        schema_info += "\nSample Data (first 1 row):\n"
        for row in sample_data[:1]:  # Just show first row to keep prompt concise
            schema_info += f"  {dict(zip([col[1] for col in columns], row))}\n"

        return schema_info

    except Exception as e:
        return f"Error getting schema: {str(e)}"


def generate_sql_query(
    question: str, schema_info: str, generation_config=None
) -> SQLQuery:
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

    @st.cache_data(ttl=15 * 60)
    def get_sql_query_raw_completion(system_prompt: str, **kwargs):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": system_prompt}],
                response_model=SQLQuery,
                **kwargs,
            ).model_dump()
            return response
        except Exception as e:
            logger.error(e)
            response = SQLQuery(
                sql="",
                explanation=f"Error generating query: {str(e)}",
                confidence=0.0,
                tables_used=[],
            ).model_dump()
            return response

    raw_completion = get_sql_query_raw_completion(
        system_prompt, generation_config=generation_config
    )
    return SQLQuery(**raw_completion)


def analyze_retrieved_data(
    df: pd.DataFrame, user_query: str = None, **kwargs
) -> QueryonData:
    if len(df) >= 20 or len(df.columns) > 5:
        st.warning("Dataset may be too large to anal")
    df_json_str: str = df.to_json(orient="records")
    if user_query is None:
        user_query = ""
    system_prompt = f""" 
You are an expert data analyst. Your task is to analyze the data provided below to 
answer a user query on the data.

## Guidelines

- If the query is relevant to the dataset and is answerable, set status to True and 
provide a response.
- If the query is not relevant to the dataset, set statust to False and explain why 
the query is not relevant to the dataset.
- If no query is provided, provide key insights about the dataset

## Dataset

{df_json_str}

## User query

{user_query}
"""

    @st.cache_data(ttl=15 * 60)
    def analyze_retrieved_data_raw_completion(system_prompt: str, **kwargs):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": system_prompt}],
                response_model=QueryonData,
                **kwargs,
            ).model_dump()
            return response
        except Exception as e:
            logger.error(e)
            response = QueryonData(
                status=False,
                answer="Error occurred while attempting to answer query on the dataset",
            ).model_dump()
            return response

    raw_completion = analyze_retrieved_data_raw_completion(system_prompt, **kwargs)
    return QueryonData(**raw_completion)


def execute_query(conn, sql: str) -> Dict[str, Any]:
    def clean_sql(sql: str) -> str:
        # Enhanced safety check
        sql_upper = sql.upper().strip()

        # Remove comments and normalize whitespace
        sql_clean = " ".join(sql_upper.split())

        # Check for dangerous keywords
        dangerous_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "REPLACE",
            "MERGE",
            "EXEC",
            "EXECUTE",
            "PRAGMA",
            "ATTACH",
            "DETACH",
        ]

        # Allow WITH clause by checking if query starts with WITH or SELECT
        if not (sql_clean.startswith("SELECT") or sql_clean.startswith("WITH")):
            return {
                "success": False,
                "error": "Only SELECT queries (optionally with WITH clauses) are allowed",
                "data": None,
            }

        # Check for dangerous keywords in the query
        for keyword in dangerous_keywords:
            # Use word boundaries to avoid false positives (e.g. "SELECT" in "SELECTED")
            if f" {keyword} " in f" {sql_clean} " or sql_clean.startswith(
                f"{keyword} "
            ):
                return {
                    "success": False,
                    "error": f"Query contains prohibited keyword: {keyword}",
                    "data": None,
                }

        # Add LIMIT clause if not present
        if "LIMIT" not in sql_upper:
            sql = f"{sql.rstrip(';')} LIMIT 1000"

        return sql

    # """Execute SQL query and return results"""
    try:
        sql = clean_sql(sql)
        # Execute query after cleanup
        df = pd.read_sql_query(sql, conn)

        return {"success": True, "data": df, "error": None, "row_count": len(df)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": None, "row_count": 0}


def main():
    st.set_page_config(page_title="Data Analysis Agent", layout="wide")

    st.title("🔍 Text-to-SQL and Data Analysis Agent")
    st.subheader("Ask questions about the Superstore dataset in natural language!")
    st.markdown(
        "#### Workflow \n\n"
        "- **Step 1** : The LLM fetches data that answers your query by converting it to SQL\n\n"
        "- **Step 2**: The LLM answers a follow-up question on the fetched data, "
        "e.g. 'Give me some key insights on the dataset'"
    )

    # Initialize database
    if "db_conn" not in st.session_state:
        with st.spinner("Setting up database..."):
            conn, columns = setup_database()
            if conn:
                st.session_state.db_conn = conn
                st.session_state.schema_info = get_schema_info(conn)
                st.success("Database loaded successfully!")
            else:
                st.error("Failed to load database")
                return

    if st.checkbox("Show dataset preview", False):
        st.write(
            pd.read_sql("SELECT * FROM superstore LIMIT 5;", st.session_state.db_conn)
        )

    generation_config = {"temperature": 0.1}
    if st.checkbox("Advanced: configure LLM parameters"):
        temperature = st.slider(
            "Select temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            help="Lower value means more deterministic outputs",
        )
        generation_config["temperature"] = temperature

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
            "What are the most discounted products?",
        ]

        for example in example_questions:
            if st.button(example, key=f"example_{example}"):
                st.session_state.user_question = example

    # Main query interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Fetch data relevant to your query")
        user_question = st.text_input(
            "Ask a question about the data:",
            value=st.session_state.get("user_question", ""),
            placeholder="e.g., What are the top selling products in California?",
        )

    with col2:
        st.write("")  # Spacing
        query_button = st.button("Generate Query", type="primary")

    if query_button and user_question:
        with st.spinner("Generating SQL query..."):
            # Generate SQL
            sql_response = generate_sql_query(
                user_question,
                st.session_state.schema_info,
                generation_config=generation_config,
            )
            st.session_state.sql_response = sql_response
            # Setting this to None since a new question is asked
            st.session_state["result_query_on_data"] = None
    else:
        sql_response = st.session_state.get("sql_response")

    if sql_response is not None:
        if sql_response.sql:
            # Display generated query
            st.markdown("##### Generated SQL Query")
            col1, col2 = st.columns([3, 1])

            with col1:
                if st.checkbox("Show generated SQL query", key="show_sql_response"):
                    st.code(sql_response.sql, language="sql")

            with col2:
                st.metric(
                    "Confidence of SQL correctness",
                    f"{sql_response.confidence:.2f}",
                    help="The LLM's own "
                    "estimate of how "
                    "correct the generated SQL query is ",
                )

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
                        numeric_cols = (
                            result["data"].select_dtypes(include=["number"]).columns
                        )
                        if len(numeric_cols) > 0 and len(result["data"]) > 1:
                            st.subheader("Quick Visualization")

                            # Try to create a simple chart
                            if len(result["data"]) <= 20:  # Only for small result sets
                                try:
                                    if len(result["data"].columns) == 2:
                                        st.bar_chart(
                                            result["data"].set_index(
                                                result["data"].columns[0]
                                            )
                                        )
                                    elif len(numeric_cols) == 1:
                                        st.bar_chart(result["data"][numeric_cols[0]])
                                except:
                                    pass  # Skip visualization if it fails
                    else:
                        st.info("Query executed successfully but returned no results.")
                else:
                    st.error(f"Query execution failed: {result['error']}")

            st.session_state["result_data"] = result

        else:
            logger.error(f"Failed to get valid structured output: {sql_response}")
            st.error("Failed to generate SQL query")

    if "result_data" in st.session_state:
        result = st.session_state.result_data
        # Analyze retrieved data
        st.subheader("Ask a question to analyze fetched data")
        query_on_data = st.text_input(
            "Ask a follow-up question on the retrieved data (key insights retrived by default)",
            value=st.session_state.get("query_on_retrieved_data", ""),
            placeholder="e.g. Give me key insights on the retrieved data",
        )
        result_query_on_data = st.session_state.get("result_query_on_data")
        if st.button("Answer follow-up query", type="primary"):
            result_query_on_data = analyze_retrieved_data(
                df=result["data"],
                user_query=query_on_data,
                generation_config=generation_config,
            )
        if result_query_on_data is not None:
            if result_query_on_data.status:
                st.markdown("#### Answer:")
                st.write(result_query_on_data.answer)
                st.markdown("#### Explanation:")
                st.write(result_query_on_data.explanation)
                st.session_state["result_query_on_data"] = result_query_on_data
            else:
                st.write(
                    f"Unable to answer the query, explanation: "
                    f"{result_query_on_data.answer}"
                )


if __name__ == "__main__":
    main()

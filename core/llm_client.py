import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import google.generativeai as genai
import instructor
import pandas as pd
import os
from loguru import logger
from models.schemas import SQLQuery, QueryonData

# Configure Gemini
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except StreamlitSecretNotFoundError:
    api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=api_key)
# Initialize the Gemini GenerativeModel client
client = instructor.from_provider(model="google/gemini-2.5-flash")


def generate_sql_query(question: str, schema_info: str,
                       generation_config=None) -> SQLQuery:
    """Generate SQL query using Gemini with Instructor"""

    system_prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries.

{schema_info}

Rules:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Use proper SQL syntax for SQLite
3. Be careful with column names - use exact names from schema
4. For date operations, use SQLite date functions
5. Always include a confidence score based on query complexity and ambiguity
6. You can reference temp tables from previous cells (cell_X_results) and JOIN them with original tables
7. When referencing previous cell results, use the exact table name: cell_1_results, cell_2_results, etc.

Example queries:
- "What are the top 5 products by sales?" -> SELECT product_name, SUM(sales) as total_sales FROM superstore GROUP BY product_name ORDER BY total_sales DESC LIMIT 5
- "Show sales by category" -> SELECT category, SUM(sales) as total_sales FROM superstore GROUP BY category ORDER BY total_sales DESC
- "Compare results from cell 1 with original data" -> SELECT a.*, b.* FROM superstore a JOIN cell_1_results b ON a.product_name = b.product_name
- "Show data from previous analysis" -> SELECT * FROM cell_2_results WHERE profit > 100

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


def analyze_retrieved_data(df: pd.DataFrame, user_query: str = None,
                           **kwargs) -> QueryonData:
    """Analyze retrieved data using LLM"""
    if len(df) >= 20 or len(df.columns) > 5:
        st.warning("Dataset may be too large to analyze")

    df_json_str: str = df.to_json(orient="records")
    if user_query is None:
        user_query = ""

    system_prompt = f""" 
You are an expert data analyst. Your task is to analyze the data provided below to 
answer a user query on the data.

## Guidelines

- If the query is relevant to the dataset and is answerable, set status to True and 
provide a response.
- If the query is not relevant to the dataset, set status to False and explain why 
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
                explanation=f"Error: {str(e)}"
            ).model_dump()
            return response

    raw_completion = analyze_retrieved_data_raw_completion(system_prompt, **kwargs)
    return QueryonData(**raw_completion)


def build_enhanced_prompt(base_prompt: str, context_data: dict = None) -> str:
    """Build enhanced prompts with additional context"""
    enhanced_prompt = base_prompt

    if context_data:
        if "previous_queries" in context_data:
            enhanced_prompt += "\n\n## Previous Queries in this Session:\n"
            for i, query in enumerate(context_data["previous_queries"], 1):
                enhanced_prompt += f"{i}. {query}\n"

        if "available_tables" in context_data:
            enhanced_prompt += "\n\n## Available Tables:\n"
            for table in context_data["available_tables"]:
                enhanced_prompt += f"- {table}\n"

    return enhanced_prompt


def validate_llm_response(response, expected_fields: list) -> bool:
    """Validate LLM response has required fields"""
    try:
        for field in expected_fields:
            if not hasattr(response, field):
                logger.warning(f"Missing field in LLM response: {field}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error validating LLM response: {e}")
        return False


def get_llm_client_info():
    """Get information about the current LLM client"""
    return {
        "model": "google/gemini-2.5-flash",
        "provider": "google",
        "client_type": "instructor",
        "api_key_configured": bool(api_key)
    }
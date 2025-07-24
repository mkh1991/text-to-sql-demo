import sqlite3
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
import streamlit as st


def get_session_schema_info(
    conn, current_cell_id: int = None, exclude_current_cell=True
):
    """Get database schema information for the session including temp tables"""
    try:
        cursor = conn.cursor()

        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_info = "## Session Database Schema:\n\n"

        # Separate original and temp tables
        original_tables = []
        temp_tables = []

        for table_name in tables:
            table_name = table_name[0]
            if table_name.startswith("cell_") and table_name.endswith("_results"):
                temp_tables.append(table_name)
            else:
                original_tables.append(table_name)

        # Show original tables first
        if original_tables:
            schema_info += "## Original Tables:\n"
            for table_name in original_tables:
                schema_info += format_table_info(
                    cursor, table_name, include_sample=True
                )

        # Show temp tables
        if temp_tables:
            schema_info += "## Computed Datasets (Temp Tables):\n"
            for table_name in temp_tables:
                # Extract cell ID from table name
                cell_id = table_name.replace("cell_", "").replace("_results", "")
                if cell_id != str(current_cell_id):
                    schema_info += f"**{table_name}** (from Cell {cell_id}):\n"
                    schema_info += format_table_info(
                        cursor, table_name, include_sample=False, indent="  "
                    )

        return schema_info

    except Exception as e:
        return f"Error getting schema: {str(e)}"


def format_table_info(
    cursor, table_name: str, include_sample: bool = False, indent: str = ""
) -> str:
    """Format table information for schema display"""
    info = f"Table Name: {table_name}\n"

    # Get column info
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    info += f"{indent}Columns:\n"
    for col in columns:
        info += f"{indent}  - {col[1]} ({col[2]})\n"

    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    info += f"{indent}Rows: {row_count}\n"

    # Add sample data if requested
    if include_sample and row_count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        sample_data = cursor.fetchall()
        if sample_data:
            info += f"{indent}Sample Data:\n"
            sample_dict = dict(zip([col[1] for col in columns], sample_data[0]))
            info += f"{indent}  {sample_dict}\n"

    info += "\n"
    return info


def execute_query(conn, sql: str) -> Dict[str, Any]:
    """Execute SQL query on session database and return results"""

    def clean_sql(sql: str) -> str:
        """Enhanced safety check for SQL queries"""
        sql_upper = sql.upper().strip()
        sql_clean = " ".join(sql_upper.split())

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

        if not (sql_clean.startswith("SELECT") or sql_clean.startswith("WITH")):
            raise Exception(
                "Only SELECT queries (optionally with WITH clauses) are allowed"
            )

        for keyword in dangerous_keywords:
            if f" {keyword} " in f" {sql_clean} " or sql_clean.startswith(
                f"{keyword} "
            ):
                raise Exception(f"Query contains prohibited keyword: {keyword}")

        if "LIMIT" not in sql_upper:
            sql = f"{sql.rstrip(';')} LIMIT 1000"

        return sql

    try:
        sql = clean_sql(sql)
        # Execute query on session database
        df = pd.read_sql_query(sql, conn)
        return {"success": True, "data": df, "error": None, "row_count": len(df)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": None, "row_count": 0}


def create_temp_table(
    conn, cell_id: int, df: pd.DataFrame, description: str = None
) -> Dict[str, Any]:
    """Create a temporary table from DataFrame results"""
    table_name = f"cell_{cell_id}_results"

    try:
        # Create table from DataFrame
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Log the creation
        logger.info(
            f"Created temp table {table_name} with {len(df)} rows, {len(df.columns)} columns"
        )

        return {
            "success": True,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
        }
    except Exception as e:
        logger.error(f"Error creating temp table {table_name}: {e}")
        return {"success": False, "error": str(e)}


def drop_temp_table(conn, cell_id: int) -> Dict[str, Any]:
    """Drop a temporary table for a specific cell"""
    table_name = f"cell_{cell_id}_results"

    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        logger.info(f"Dropped temp table {table_name}")
        return {"success": True, "table_name": table_name}
    except Exception as e:
        logger.error(f"Error dropping temp table {table_name}: {e}")
        return {"success": False, "error": str(e)}


def get_temp_table_name(cell_id: int) -> str:
    """Get the temp table name for a cell ID"""
    return f"cell_{cell_id}_results"


def temp_table_exists(conn, cell_id: int) -> bool:
    """Check if a temp table exists for a cell"""
    table_name = get_temp_table_name(cell_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None
    except Exception as e:
        logger.error(f"Error checking temp table existence: {e}")
        return False


def get_table_info(conn, table_name: str):
    """Get information about a specific table"""
    try:
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if not cursor.fetchone():
            return {"exists": False}

        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        return {
            "exists": True,
            "columns": [{"name": col[1], "type": col[2]} for col in columns],
            "column_names": [col[1] for col in columns],
            "row_count": row_count,
        }

    except Exception as e:
        logger.error(f"Error getting table info for {table_name}: {e}")
        return {"exists": False, "error": str(e)}


def list_all_tables(conn) -> Dict[str, Any]:
    """List all tables in the database with their info"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]

        tables_info = {
            "original_tables": [],
            "temp_tables": [],
            "total_count": len(table_names),
        }

        for table_name in table_names:
            table_info = get_table_info(conn, table_name)
            table_data = {
                "name": table_name,
                "row_count": table_info.get("row_count", 0),
                "column_count": len(table_info.get("columns", [])),
                "columns": table_info.get("column_names", []),
            }

            if table_name.startswith("cell_") and table_name.endswith("_results"):
                # Extract cell ID
                cell_id = table_name.replace("cell_", "").replace("_results", "")
                table_data["cell_id"] = int(cell_id) if cell_id.isdigit() else None
                tables_info["temp_tables"].append(table_data)
            else:
                tables_info["original_tables"].append(table_data)

        return {"success": True, **tables_info}

    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {"success": False, "error": str(e)}


def cleanup_orphaned_temp_tables(conn, existing_cell_ids: List[int]):
    """Clean up temp tables that don't have corresponding cells"""
    try:
        tables_info = list_all_tables(conn)
        if not tables_info["success"]:
            return

        cleaned_count = 0
        for table_info in tables_info["temp_tables"]:
            cell_id = table_info.get("cell_id")
            if cell_id and cell_id not in existing_cell_ids:
                result = drop_temp_table(conn, cell_id)
                if result["success"]:
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} orphaned temp tables")

    except Exception as e:
        logger.error(f"Error cleaning up orphaned temp tables: {e}")


def get_enhanced_workspace_context(cell_id: int = None) -> str:
    """Get enhanced workspace context including all tables"""
    conn = st.session_state.session_db_conn
    try:
        #     tables_info = list_all_tables(conn)
        #
        #     context = "## Available Tables for Querying:\n\n"
        #
        #     # Original tables
        #     if tables_info.get("original_tables"):
        #         context += "### Original Database Tables:\n"
        #         for table in tables_info["original_tables"]:
        #             context += f"- **{table['name']}** ({table['row_count']} rows, {table['column_count']} columns)\n"
        #             context += f"  Columns: {', '.join(table['columns'])}\n\n"
        #
        #     # Temp tables from previous cells
        #     if tables_info.get("temp_tables"):
        #         context += "### Computed Datasets (Available for JOIN/reference):\n"
        #         for table in tables_info["temp_tables"]:
        #             cell_id = table.get("cell_id", "?")
        #             context += f"- **{table['name']}** (Cell {cell_id} results: {table['row_count']} rows, {table['column_count']} columns)\n"
        #             context += f"  Columns: {', '.join(table['columns'])}\n\n"
        context = get_session_schema_info(conn, current_cell_id=cell_id)
        prev_cell_id_str = cell_id - 1 if cell_id else ""
        context += f"## Previous Cell ID: {prev_cell_id_str}\n"
        context += "## Query Examples:\n"
        context += "- Query original data: `SELECT * FROM superstore WHERE category = 'Technology'`\n"
        context += "- Reference previous results: `SELECT * FROM cell_1_results`\n"
        context += "- Join datasets: `SELECT a.*, b.profit FROM superstore a JOIN cell_2_results b ON a.order_id = b.order_id`\n"

        return context

    except Exception as e:
        logger.error(f"Error building workspace context: {e}")
        return "Error loading workspace context"

import streamlit as st
import pandas as pd
from loguru import logger
from core.database import get_session_schema_info, execute_query, \
    get_enhanced_workspace_context, create_temp_table
from core.llm_client import generate_sql_query
from core.session_manager import reset_session, clear_session_data
from utils.helpers import get_cell_status_emoji, format_row_count, truncate_text


def initialize_notebook():
    """Initialize the notebook cell system in session state"""
    if "notebook_cells" not in st.session_state:
        st.session_state.notebook_cells = []
    if "cell_counter" not in st.session_state:
        st.session_state.cell_counter = 0


def add_new_cell():
    """Add a new empty cell to the notebook"""
    st.session_state.cell_counter += 1
    new_cell = {
        "id": st.session_state.cell_counter,
        "query": "",
        "sql_response": None,
        "execution_result": None,
        "analysis_result": None,
        "status": "empty"  # empty, running, complete, error
    }
    st.session_state.notebook_cells.append(new_cell)
    return new_cell


def delete_cell(cell_id):
    """Delete a cell by ID"""
    # Find and remove the cell
    st.session_state.notebook_cells = [
        cell for cell in st.session_state.notebook_cells
        if cell["id"] != cell_id
    ]

    # TODO: In Phase 2, we'll also need to:
    # 1. Drop the corresponding temp table from the database
    # 2. Update any dependent cells that reference this cell's data

    logger.info(f"Deleted cell {cell_id}")


def execute_cell(cell, generation_config):
    """Execute a single cell using session database"""
    cell["status"] = "running"

    try:
        # Build workspace context
        workspace_context = get_enhanced_workspace_context(cell_id=cell["id"])
        logger.info(f"Current workspace context: \n\n{workspace_context}")

        with st.spinner("Generating SQL query..."):
            sql_response = generate_sql_query(
                cell["query"],
                schema_info=workspace_context,
                generation_config=generation_config
            )
            cell["sql_response"] = sql_response

        if sql_response.sql:
            with st.spinner("Executing query..."):
                # Use session database connection
                result = execute_query(st.session_state.session_db_conn,
                                       sql_response.sql)
                cell["execution_result"] = result

                if result["success"]:
                    if result["row_count"] > 0:
                        temp_result = create_temp_table(
                            st.session_state.session_db_conn,  # Database connection
                            cell["id"],  # Cell ID (becomes cell_X_results)
                            result["data"],  # DataFrame from query results
                            description=cell["query"]  # Optional description
                        )
                        if temp_result["success"]:
                            cell["temp_table_name"] = temp_result["table_name"]
                            logger.info(
                                f"Created temp table {temp_result['table_name']} for cell {cell['id']}")
                            # Update schema info to include new temp table
                            st.session_state.session_schema_info = get_session_schema_info(
                                st.session_state.session_db_conn
                            )
                    cell["status"] = "complete"
                    # TODO: Phase 2 - Create temp table from results
                    # create_temp_table(conn, f"cell_{cell['id']}_results", result["data"])
                else:
                    cell["status"] = "error"
        else:
            cell["status"] = "error"

    except Exception as e:
        cell["status"] = "error"
        cell["execution_result"] = {"success": False, "error": str(e)}


# UI RENDERING FUNCTIONS

def render_sidebar():
    """Render the complete sidebar"""
    with st.sidebar:
        render_session_info()
        render_database_schema()
        render_workspace_datasets()
        render_session_actions()
        render_llm_settings()


def render_session_info():
    """Render session information section"""
    st.header("🔧 Session Info")
    session_id = st.session_state.get("session_id", "Unknown")
    st.info(f"**Session ID:** `{session_id}`")


def render_database_schema():
    """Render database schema information"""
    with st.expander("Session Database Schema", expanded=False):
        schema_info = st.session_state.get("session_schema_info", "Schema not loaded")
        st.text(schema_info)


def render_workspace_datasets():
    """Render workspace datasets information"""
    if not st.session_state.get("notebook_cells"):
        return

    completed_cells = [
        cell for cell in st.session_state.notebook_cells
        if cell["execution_result"] and cell["execution_result"]["success"]
    ]

    if not completed_cells:
        return

    st.header("📋 Computed Datasets")

    for cell in completed_cells:
        df = cell["execution_result"]["data"]
        with st.expander(f"Cell {cell['id']} ({format_row_count(len(df))} rows)",
                         expanded=False):
            st.write(f"**Query:** {truncate_text(cell['query'], 50)}")
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            st.dataframe(df.head(3), use_container_width=True)


def render_session_actions():
    """Render session action buttons"""
    st.header("🗂️ Session Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 New Session", help="Start a fresh session"):
            reset_session()
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Data", help="Clear all session data"):
            clear_session_data()
            st.rerun()


def render_llm_settings():
    """Render LLM configuration settings"""
    st.header("⚙️ Settings")
    generation_config = {"temperature": 0.1}

    if st.checkbox("Advanced: configure LLM parameters"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            help="Lower value means more deterministic outputs"
        )
        generation_config["temperature"] = temperature

    # Store in session state for access by cells
    st.session_state.generation_config = generation_config

    return generation_config


def render_cell(cell, cell_index, generation_config):
    """Render a single notebook cell"""

    # Cell container with status indicator
    with st.container():
        # Cell header with delete button and status
        col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
        with col_header1:
            status_emoji = get_cell_status_emoji(cell["status"])
            st.markdown(f"###### {status_emoji} [{cell['id']}]")
        with col_header2:
            if cell["status"] == "running":
                st.info("🔄 Running...")
            elif cell["status"] == "complete":
                st.success("✅ Complete")
            elif cell["status"] == "error":
                st.error("❌ Error")
        with col_header3:
            if st.button("🗑️", key=f"delete_cell_{cell['id']}",
                         help="Delete this cell"):
                delete_cell(cell['id'])
                st.rerun()

        # Query input
        cell_key = f"cell_{cell['id']}"
        new_query = st.text_input(
            f"Query for Cell {cell['id']}:",
            value=cell["query"],
            key=f"{cell_key}_input",
            placeholder="Ask a question about the data or reference previous cells..."
        )

        # Update cell query if changed
        if new_query != cell["query"]:
            cell["query"] = new_query

        # Execute button
        col1, col2 = st.columns([1, 4])
        with col1:
            execute_button = st.button(
                "▶️ Execute",
                key=f"{cell_key}_execute",
                type="primary" if cell["status"] == "empty" else "secondary",
                disabled=(cell["status"] == "running")
            )

        # Execute cell
        if execute_button and cell["query"]:
            execute_cell(cell, generation_config)
            st.rerun()

        # Show results if available
        render_cell_results(cell)

        st.markdown("---")


def render_cell_results(cell):
    """Render cell execution results"""

    # Show SQL query if available
    if cell["sql_response"] and cell["sql_response"].sql:
        with st.expander("🔍 Generated SQL", expanded=False):
            st.code(cell["sql_response"].sql, language="sql")
            st.write(f"**Explanation:** {cell['sql_response'].explanation}")
            st.metric("Confidence", f"{cell['sql_response'].confidence:.2f}")

    # Show execution results
    if cell["execution_result"]:
        if cell["execution_result"]["success"]:
            df = cell["execution_result"]["data"]
            st.write(
                f"**Results:** {format_row_count(cell['execution_result']['row_count'])} rows")

            if cell["execution_result"]["row_count"] > 0:
                st.dataframe(df, use_container_width=True)

                # Simple visualization for small datasets
                render_simple_visualization(df)
        else:
            st.error(f"Execution failed: {cell['execution_result']['error']}")


def render_simple_visualization(df):
    """Render simple visualization for small datasets"""
    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0 and len(df) <= 20 and len(df) > 1:
        try:
            if len(df.columns) == 2:
                st.subheader("📊 Quick Visualization")
                st.bar_chart(df.set_index(df.columns[0]))
        except Exception as e:
            # Silently fail visualization - not critical
            logger.debug(f"Visualization failed: {e}")


def render_notebook_interface():
    """Render the main notebook interface"""
    # Initialize notebook if needed
    initialize_notebook()

    # Render sidebar
    render_sidebar()

    # Get generation config from sidebar
    generation_config = st.session_state.get("generation_config", {"temperature": 0.1})

    # Show database preview option
    if st.checkbox("Show original dataset preview", False):
        preview_df = pd.read_sql("SELECT * FROM superstore LIMIT 5;",
                                 st.session_state.session_db_conn)
        st.dataframe(preview_df, use_container_width=True)

    # Main notebook interface
    st.header("🔬 Analysis Cells")

    # If no cells exist, create the first one
    if not st.session_state.notebook_cells:
        add_new_cell()

    # Show cell count and management info
    if len(st.session_state.notebook_cells) > 1:
        st.info(f"📊 **{len(st.session_state.notebook_cells)} cells** in this notebook")

    # Render all cells
    for i, cell in enumerate(st.session_state.notebook_cells):
        render_cell(cell, i, generation_config)

    # Add new cell button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➕ Add New Cell", type="primary", use_container_width=True):
            add_new_cell()
            st.rerun()
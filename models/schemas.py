from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SQLQuery(BaseModel):
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Plain English explanation of what the query does"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    tables_used: List[str] = Field(description="List of tables used in the query")


class QueryonData(BaseModel):
    status: bool = Field(
        description="Whether the query is relevant to the dataset and is answerable"
    )
    answer: str = Field(
        description="Detailed answer to the query, using multiple "
        "bullet points if necessary"
    )
    explanation: str = Field(
        description="Explanation for the answer or for why the query is not "
        "answerable for this dataset. NEVER answer the query if it is not relevant to the dataset."
    )


class CellData(BaseModel):
    """Data model for notebook cells (Phase 2 enhancement)"""
    id: int = Field(description="Unique cell identifier")
    query: str = Field(description="User's natural language query")
    sql_response: Optional[SQLQuery] = Field(default=None, description="Generated SQL query")
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Query execution results")
    analysis_result: Optional[QueryonData] = Field(default=None, description="Data analysis results")
    status: str = Field(default="empty", description="Cell execution status")
    temp_table_name: Optional[str] = Field(default=None, description="Associated temporary table name")
    created_at: Optional[str] = Field(default=None, description="Cell creation timestamp")
    executed_at: Optional[str] = Field(default=None, description="Last execution timestamp")


class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str = Field(description="Unique session identifier")
    created_at: str = Field(description="Session creation timestamp")
    db_path: str = Field(description="Path to session database file")
    cell_count: int = Field(description="Number of cells in the notebook")
    table_count: int = Field(description="Number of tables in session database")


class WorkspaceContext(BaseModel):
    """Context information for workspace"""
    original_tables: List[str] = Field(description="Original database tables")
    temp_tables: List[str] = Field(description="Temporary tables from cell executions")
    completed_cells: List[int] = Field(description="List of completed cell IDs")
    total_rows_processed: int = Field(description="Total rows across all datasets")


class DatabaseTable(BaseModel):
    """Database table information"""
    name: str = Field(description="Table name")
    row_count: int = Field(description="Number of rows")
    column_count: int = Field(description="Number of columns")
    columns: List[str] = Field(description="List of column names")
    table_type: str = Field(description="Type: 'original' or 'temp'")
    source_cell_id: Optional[int] = Field(default=None, description="Cell ID if temp table")


class ExecutionResult(BaseModel):
    """Standardized execution result"""
    success: bool = Field(description="Whether execution was successful")
    data: Optional[Any] = Field(default=None, description="Result data (usually DataFrame)")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    row_count: int = Field(default=0, description="Number of rows returned")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")


class LLMConfig(BaseModel):
    """LLM configuration model"""
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    model: str = Field(default="google/gemini-2.5-flash", description="Model identifier")
    timeout: int = Field(default=30, description="Request timeout in seconds")
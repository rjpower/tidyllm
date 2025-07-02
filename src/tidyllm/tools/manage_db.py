"""Database management tools."""

from typing import Any

from pydantic import BaseModel, Field

from tidyllm.cli import multi_cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


# Query Database Tool
class DBQueryArgs(BaseModel):
    """Arguments for database queries."""
    sql: str = Field(description="SQL SELECT query")
    params: dict[str, Any] = Field(default_factory=dict, description="Query parameters")


class DBQueryResult(BaseModel):
    """Result of database query."""
    success: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    error: str | None = None


@register()
def db_query(args: DBQueryArgs) -> DBQueryResult:
    """Execute SELECT queries safely.
    
    Example usage: db_query({"sql": "SELECT * FROM vocab WHERE word LIKE ?", "params": {"%hello%"}})
    """
    ctx = get_tool_context()
    db = ctx.db

    try:
        # Only allow SELECT queries
        if not args.sql.strip().upper().startswith("SELECT"):
            return DBQueryResult(success=False, error="Only SELECT queries are allowed")

        # Execute with parameters if provided
        params = list(args.params.values()) if args.params else None
        cursor = db.query(args.sql, params)

        result_rows = [dict(row.items()) for row in cursor]

        return DBQueryResult(success=True, rows=result_rows, count=len(result_rows))

    except Exception as e:
        return DBQueryResult(success=False, error=f"Database error: {str(e)}")


# Execute Database Statements Tool
class DBExecuteArgs(BaseModel):
    """Arguments for database execution."""
    sql: str = Field(description="SQL statement (INSERT, UPDATE, DELETE)")
    params: dict[str, Any] = Field(default_factory=dict, description="Statement parameters")


class DBExecuteResult(BaseModel):
    """Result of database execution."""
    success: bool
    affected_count: int = 0
    error: str | None = None


@register()
def db_execute(args: DBExecuteArgs) -> DBExecuteResult:
    """Execute INSERT, UPDATE, DELETE statements safely.
    
    Example usage: db_execute({"sql": "INSERT INTO vocab (word, translation) VALUES (?, ?)", "params": {"hello": "hola"}})
    """
    ctx = get_tool_context()
    db = ctx.db

    try:
        # Disallow dangerous operations
        sql_upper = args.sql.strip().upper()
        if any(keyword in sql_upper for keyword in ["DROP", "TRUNCATE", "ALTER TABLE"]):
            return DBExecuteResult(success=False, error="Dangerous operations are not allowed")

        # Execute with parameters if provided
        params = list(args.params.values()) if args.params else None
        affected_count = db.mutate(args.sql, params)

        return DBExecuteResult(success=True, affected_count=affected_count)

    except Exception as e:
        return DBExecuteResult(success=False, error=f"Database error: {str(e)}")


# List Tables Tool
class DBListTablesArgs(BaseModel):
    """Arguments for listing tables."""
    pass  # No arguments needed


class DBListTablesResult(BaseModel):
    """Result of listing tables."""
    success: bool
    tables: list[str] = Field(default_factory=list)
    count: int = 0


@register()
def db_list_tables(args: DBListTablesArgs) -> DBListTablesResult:
    """List all tables in the database.
    
    Example usage: db_list_tables({})
    """
    del args  # Unused parameter
    ctx = get_tool_context()
    db = ctx.db

    try:
        cursor = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor]

        return DBListTablesResult(success=True, tables=tables, count=len(tables))

    except Exception:
        return DBListTablesResult(success=False, tables=[])


# Get Schema Tool
class DBSchemaArgs(BaseModel):
    """Arguments for getting schema."""
    table: str | None = Field(None, description="Specific table name (all tables if not provided)")


class DBSchemaResult(BaseModel):
    """Result of getting schema."""
    success: bool
    db_schema: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    error: str | None = None


@register()
def db_schema(args: DBSchemaArgs) -> DBSchemaResult:
    """Get database schema information.
    
    Example usage: db_schema({"table": "vocab"}) or db_schema({}) for all tables
    """
    ctx = get_tool_context()
    db = ctx.db

    try:
        # Use the built-in schema method
        full_schema = db.schema()

        # Filter by table if specified
        if args.table:
            table_schema = full_schema.get_table(args.table)
            if not table_schema:
                return DBSchemaResult(
                    success=False, error=f"Table '{args.table}' not found"
                )
            tables_to_process = [table_schema]
        else:
            tables_to_process = full_schema.tables

        schema = {}
        for table in tables_to_process:
            schema[table.name] = [
                {
                    "name": col.name,
                    "type": col.type,
                    "nullable": str(not col.not_null),
                    "default": col.default or "",
                    "primary_key": str(col.primary_key),
                }
                for col in table.columns
            ]

        return DBSchemaResult(success=True, db_schema=schema)

    except Exception as e:
        return DBSchemaResult(success=False, error=f"Database error: {str(e)}")


if __name__ == "__main__":
    multi_cli_main(
        [db_query, db_execute, db_list_tables, db_schema],
        default_function="db_list_tables",
        context_cls=ToolContext,
    )

"""Database management tools."""

from typing import Any

from pydantic import BaseModel

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


@register()
def db_query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Execute SELECT queries safely.
    
    Args:
        sql: SQL SELECT query
        params: Query parameters as list (optional)
        
    Example usage: db_query("SELECT * FROM vocab WHERE word LIKE ?", ["%hello%"])
    """
    ctx = get_tool_context()
    db = ctx.db

    # Only allow SELECT queries
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    # Execute with parameters if provided
    cursor = db.query(sql, params)

    return [dict(row.items()) for row in cursor]


class DBExecuteResult(BaseModel):
    """Result of database execution."""
    affected_count: int


@register()
def db_execute(sql: str, params: list[Any] | None = None) -> DBExecuteResult:
    """Execute INSERT, UPDATE, DELETE statements safely.
    
    Args:
        sql: SQL statement (INSERT, UPDATE, DELETE)
        params: Statement parameters as list (optional)
        
    Example usage: db_execute("INSERT INTO vocab (word, translation) VALUES (?, ?)", ["hello", "hola"])
    """
    ctx = get_tool_context()
    db = ctx.db

    # Disallow dangerous operations
    sql_upper = sql.strip().upper()
    if any(keyword in sql_upper for keyword in ["DROP", "TRUNCATE", "ALTER TABLE"]):
        raise ValueError("Dangerous operations are not allowed")

    # Execute with parameters if provided
    affected_count = db.mutate(sql, params)
    return DBExecuteResult(affected_count=affected_count)


@register()
def db_schema(table: str | None = None) -> dict[str, list[dict[str, str]]]:
    """Get database schema information.
    
    Args:
        table: Specific table name (all tables if not provided)
        
    Example usage: db_schema("vocab") or db_schema() for all tables
    """
    ctx = get_tool_context()
    db = ctx.db

    # Use the built-in schema method
    full_schema = db.schema()

    # Filter by table if specified
    if table:
        table_schema = full_schema.get_table(table)
        if not table_schema:
            raise ValueError(f"Table '{table}' not found")
        tables_to_process = [table_schema]
    else:
        tables_to_process = full_schema.tables

    schema = {}
    for table_obj in tables_to_process:
        schema[table_obj.name] = [
            {
                "name": col.name,
                "type": col.type,
                "nullable": str(not col.not_null),
                "default": col.default or "",
                "primary_key": str(col.primary_key),
            }
            for col in table_obj.columns
        ]

    return schema


if __name__ == "__main__":
    cli_main(
        [db_query, db_execute, db_schema],
        default_function="db_schema",
        context_cls=ToolContext,
    )

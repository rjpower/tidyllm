"""Database management tools."""

from typing import Any

from pydantic import BaseModel, Field

from tidyllm.context import get_tool_context
from tidyllm.multi_cli import simple_cli_main
from tidyllm.registry import register
from tidyllm.tools.db import init_database, row_to_dict


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


@register
def db_query(args: DBQueryArgs) -> DBQueryResult:
    """Execute SELECT queries safely.
    
    Example usage: db_query({"sql": "SELECT * FROM vocab WHERE word LIKE ?", "params": {"%hello%"}})
    """
    ctx = get_tool_context()
    init_database(ctx)
    
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Only allow SELECT queries
        if not args.sql.strip().upper().startswith("SELECT"):
            return DBQueryResult(success=False, error="Only SELECT queries are allowed")
            
        # Execute with parameters if provided
        if args.params:
            cursor.execute(args.sql, list(args.params.values()))
        else:
            cursor.execute(args.sql)
            
        rows = cursor.fetchall()
        result_rows = [row_to_dict(row) for row in rows]
        
        return DBQueryResult(success=True, rows=result_rows, count=len(result_rows))
        
    except Exception as e:
        return DBQueryResult(success=False, error=f"Database error: {str(e)}")
    finally:
        conn.close()


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


@register
def db_execute(args: DBExecuteArgs) -> DBExecuteResult:
    """Execute INSERT, UPDATE, DELETE statements safely.
    
    Example usage: db_execute({"sql": "INSERT INTO vocab (word, translation) VALUES (?, ?)", "params": {"hello": "hola"}})
    """
    ctx = get_tool_context()
    init_database(ctx)
    
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Disallow dangerous operations
        sql_upper = args.sql.strip().upper()
        if any(keyword in sql_upper for keyword in ["DROP", "TRUNCATE", "ALTER TABLE"]):
            return DBExecuteResult(success=False, error="Dangerous operations are not allowed")
            
        # Execute with parameters if provided
        if args.params:
            cursor.execute(args.sql, list(args.params.values()))
        else:
            cursor.execute(args.sql)
            
        conn.commit()
        
        return DBExecuteResult(success=True, affected_count=cursor.rowcount)
        
    except Exception as e:
        conn.rollback()
        return DBExecuteResult(success=False, error=f"Database error: {str(e)}")
    finally:
        conn.close()


# List Tables Tool
class DBListTablesArgs(BaseModel):
    """Arguments for listing tables."""
    pass  # No arguments needed


class DBListTablesResult(BaseModel):
    """Result of listing tables."""
    success: bool
    tables: list[str] = Field(default_factory=list)
    count: int = 0


@register
def db_list_tables(args: DBListTablesArgs) -> DBListTablesResult:
    """List all tables in the database.
    
    Example usage: db_list_tables({})
    """
    ctx = get_tool_context()
    init_database(ctx)
    
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = cursor.fetchall()
        tables = [row["name"] for row in rows]
        
        return DBListTablesResult(success=True, tables=tables, count=len(tables))
        
    except Exception:
        return DBListTablesResult(success=False, tables=[])
    finally:
        conn.close()


# Get Schema Tool
class DBSchemaArgs(BaseModel):
    """Arguments for getting schema."""
    table: str | None = Field(None, description="Specific table name (all tables if not provided)")


class DBSchemaResult(BaseModel):
    """Result of getting schema."""
    success: bool
    db_schema: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    error: str | None = None


@register
def db_schema(args: DBSchemaArgs) -> DBSchemaResult:
    """Get database schema information.
    
    Example usage: db_schema({"table": "vocab"}) or db_schema({}) for all tables
    """
    ctx = get_tool_context()
    init_database(ctx)
    
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get specific table or all tables
        if args.table:
            tables_to_check = [{"name": args.table}]
        else:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables_to_check = cursor.fetchall()
        
        schema = {}
        for table_row in tables_to_check:
            table_name = table_row["name"]
            
            # Get column info for each table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            if columns:  # Only add if table exists and has columns
                schema[table_name] = [
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "nullable": str(not col["notnull"]),
                        "default": str(col["dflt_value"]) if col["dflt_value"] is not None else "",
                        "primary_key": str(bool(col["pk"]))
                    }
                    for col in columns
                ]
            
        return DBSchemaResult(success=True, db_schema=schema)
        
    except Exception as e:
        return DBSchemaResult(success=False, error=f"Database error: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    simple_cli_main([db_query, db_execute, db_list_tables, db_schema], default_function="db_list_tables")
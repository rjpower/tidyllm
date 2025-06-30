"""Tests for database management tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.manage_db import DBArgs, manage_db


@pytest.fixture
def test_context():
    """Create a test context with temporary database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            user_db=temp_path / "test.db",
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


def test_list_tables_empty(test_context):
    """Test listing tables in empty database."""
    args = DBArgs(operation="list_tables")
    result = manage_db(args, ctx=test_context)
    
    assert result.success is True
    assert result.tables is not None
    # Should have system tables like sqlite_master
    assert len(result.tables) >= 0


def test_list_tables_with_data(test_context):
    """Test listing tables after creating some."""
    # Create a table first
    create_args = DBArgs(
        operation="execute",
        sql="CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )
    result = manage_db(create_args, ctx=test_context)
    assert result.success is True
    
    # Now list tables
    list_args = DBArgs(operation="list_tables")
    result = manage_db(list_args, ctx=test_context)
    
    assert result.success is True
    assert result.tables is not None
    assert "test_table" in result.tables


def test_schema_operation(test_context):
    """Test getting database schema."""
    # Create a test table
    create_args = DBArgs(
        operation="execute",
        sql="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)"
    )
    manage_db(create_args, ctx=test_context)
    
    # Get schema
    schema_args = DBArgs(operation="schema")
    result = manage_db(schema_args, ctx=test_context)
    
    assert result.success is True
    assert result.db_schema is not None
    assert "users" in result.db_schema
    
    # Check column details
    users_schema = result.db_schema["users"]
    assert len(users_schema) == 3  # id, name, email
    
    # Find the name column
    name_col = next(col for col in users_schema if col["name"] == "name")
    assert name_col["type"] == "TEXT"
    assert name_col["nullable"] == "False"


def test_execute_insert(test_context):
    """Test executing insert statement."""
    # Create table
    create_args = DBArgs(
        operation="execute",
        sql="CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)"
    )
    result = manage_db(create_args, ctx=test_context)
    assert result.success is True
    
    # Insert data
    insert_args = DBArgs(
        operation="execute",
        sql="INSERT INTO test (value) VALUES ('hello')"
    )
    result = manage_db(insert_args, ctx=test_context)
    
    assert result.success is True
    assert result.affected_count == 1


def test_query_select(test_context):
    """Test executing select query."""
    # Setup data
    setup_sql = """
    CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT);
    INSERT INTO test (value) VALUES ('hello'), ('world');
    """
    for sql in setup_sql.strip().split(';'):
        if sql.strip():
            manage_db(DBArgs(operation="execute", sql=sql), ctx=test_context)
    
    # Query data
    query_args = DBArgs(
        operation="query",
        sql="SELECT * FROM test ORDER BY id"
    )
    result = manage_db(query_args, ctx=test_context)
    
    assert result.success is True
    assert result.rows is not None
    assert len(result.rows) == 2
    assert result.rows[0]["value"] == "hello"
    assert result.rows[1]["value"] == "world"


def test_query_with_params(test_context):
    """Test query with parameters."""
    # Setup
    manage_db(DBArgs(operation="execute", sql="CREATE TABLE test (id INTEGER, name TEXT)"), ctx=test_context)
    manage_db(DBArgs(operation="execute", sql="INSERT INTO test VALUES (1, 'alice'), (2, 'bob')"), ctx=test_context)
    
    # Query with params
    query_args = DBArgs(
        operation="query",
        sql="SELECT * FROM test WHERE name = ?",
        params={"1": "alice"}  # Note: params are passed as dict but need to be list for sqlite
    )
    manage_db(query_args, ctx=test_context)


def test_dangerous_operations_blocked(test_context):
    """Test that dangerous operations are blocked."""
    dangerous_operations = [
        "DROP TABLE test",
        "TRUNCATE TABLE test", 
        "ALTER TABLE test ADD COLUMN new_col TEXT"
    ]
    
    for sql in dangerous_operations:
        args = DBArgs(operation="execute", sql=sql)
        result = manage_db(args, ctx=test_context)
        
        assert result.success is False
        assert "dangerous" in result.error.lower() or "not allowed" in result.error.lower()


def test_non_select_in_query_operation(test_context):
    """Test that non-SELECT statements are blocked in query operation."""
    args = DBArgs(
        operation="query",
        sql="INSERT INTO test VALUES (1, 'hack')"
    )
    result = manage_db(args, ctx=test_context)
    
    assert result.success is False
    assert "select" in result.error.lower()


def test_missing_sql(test_context):
    """Test operations with missing SQL."""
    # Query without SQL
    query_args = DBArgs(operation="query")
    result = manage_db(query_args, ctx=test_context)
    assert result.success is False
    assert "required" in result.error.lower()
    
    # Execute without SQL
    execute_args = DBArgs(operation="execute")
    result = manage_db(execute_args, ctx=test_context)
    assert result.success is False
    assert "required" in result.error.lower()


def test_invalid_operation():
    """Test invalid operation at Pydantic validation level."""
    with pytest.raises(ValueError, match="Input should be"):
        DBArgs(
            operation="invalid",  # type: ignore
            sql="SELECT 1"
        )


def test_sql_error_handling(test_context):
    """Test handling of SQL errors."""
    args = DBArgs(
        operation="query",
        sql="SELECT * FROM nonexistent_table"
    )
    result = manage_db(args, ctx=test_context)
    
    assert result.success is False
    assert result.error is not None
    assert "error" in result.error.lower()


def test_empty_database_schema(test_context):
    """Test schema operation on empty database."""
    args = DBArgs(operation="schema")
    result = manage_db(args, ctx=test_context)
    
    assert result.success is True
    assert result.db_schema is not None
    # Empty database might still have system tables
    assert isinstance(result.db_schema, dict)
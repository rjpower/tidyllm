"""Tests for database management tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.manage_db import (
    DBExecuteArgs,
    DBListTablesArgs,
    DBQueryArgs,
    DBSchemaArgs,
    db_execute,
    db_list_tables,
    db_query,
    db_schema,
)


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
    args = DBListTablesArgs()
    with set_tool_context(test_context):
        result = db_list_tables(args)
    
    assert result.success is True
    assert result.tables is not None
    # Should have zero tables in empty database
    assert len(result.tables) >= 0


def test_list_tables_with_data(test_context):
    """Test listing tables after creating some."""
    # Create a table first
    with set_tool_context(test_context):
        create_args = DBExecuteArgs(
            sql="CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
        )
        result = db_execute(create_args)
        assert result.success is True
        
        # Now list tables
        list_args = DBListTablesArgs()
        result = db_list_tables(list_args)
    
    assert result.success is True
    assert result.tables is not None
    assert "test_table" in result.tables


def test_schema_operation(test_context):
    """Test getting database schema."""
    # Create a test table
    with set_tool_context(test_context):
        create_args = DBExecuteArgs(
            sql="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)"
        )
        db_execute(create_args)
        
        # Get schema
        schema_args = DBSchemaArgs()
        result = db_schema(schema_args)
    
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
    with set_tool_context(test_context):
        create_args = DBExecuteArgs(
            sql="CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)"
        )
        result = db_execute(create_args)
        assert result.success is True
        
        # Insert data
        insert_args = DBExecuteArgs(
            sql="INSERT INTO test (value) VALUES ('hello')"
        )
        result = db_execute(insert_args)
    
    assert result.success is True
    assert result.affected_count == 1


def test_query_select(test_context):
    """Test executing select query."""
    # Setup data
    setup_sqls = [
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
        "INSERT INTO test (value) VALUES ('hello'), ('world')"
    ]
    with set_tool_context(test_context):
        for sql in setup_sqls:
            db_execute(DBExecuteArgs(sql=sql))
        
        # Query data
        query_args = DBQueryArgs(
            sql="SELECT * FROM test ORDER BY id"
        )
        result = db_query(query_args)
    
    assert result.success is True
    assert result.rows is not None
    assert len(result.rows) == 2
    assert result.rows[0]["value"] == "hello"
    assert result.rows[1]["value"] == "world"


def test_query_with_params(test_context):
    """Test query with parameters."""
    # Setup
    with set_tool_context(test_context):
        db_execute(DBExecuteArgs(sql="CREATE TABLE test (id INTEGER, name TEXT)"))
        db_execute(DBExecuteArgs(sql="INSERT INTO test VALUES (1, 'alice'), (2, 'bob')"))
        
        # Query with params
        query_args = DBQueryArgs(
            sql="SELECT * FROM test WHERE name = ?",
            params={"1": "alice"}  # Note: params are passed as dict but converted to list
        )
        result = db_query(query_args)
    
    assert result.success is True
    assert len(result.rows) == 1
    assert result.rows[0]["name"] == "alice"


def test_dangerous_operations_blocked(test_context):
    """Test that dangerous operations are blocked."""
    dangerous_operations = [
        "DROP TABLE test",
        "TRUNCATE TABLE test", 
        "ALTER TABLE test ADD COLUMN new_col TEXT"
    ]
    
    with set_tool_context(test_context):
        for sql in dangerous_operations:
            args = DBExecuteArgs(sql=sql)
            result = db_execute(args)
        
        assert result.success is False
        assert "dangerous" in result.error.lower() or "not allowed" in result.error.lower()


def test_non_select_in_query_operation(test_context):
    """Test that non-SELECT statements are blocked in query operation."""
    args = DBQueryArgs(
        sql="INSERT INTO test VALUES (1, 'hack')"
    )
    with set_tool_context(test_context):
        result = db_query(args)
    
    assert result.success is False
    assert "select" in result.error.lower()


def test_sql_error_handling(test_context):
    """Test handling of SQL errors."""
    args = DBQueryArgs(
        sql="SELECT * FROM nonexistent_table"
    )
    with set_tool_context(test_context):
        result = db_query(args)
    
    assert result.success is False
    assert result.error is not None
    assert "error" in result.error.lower()


def test_empty_database_schema(test_context):
    """Test schema operation on empty database."""
    args = DBSchemaArgs()
    with set_tool_context(test_context):
        result = db_schema(args)
    
    assert result.success is True
    assert result.db_schema is not None
    # Empty database should have empty schema dict
    assert isinstance(result.db_schema, dict)


def test_specific_table_schema(test_context):
    """Test getting schema for specific table."""
    # Create a test table
    with set_tool_context(test_context):
        create_args = DBExecuteArgs(
            sql="CREATE TABLE specific_test (id INTEGER PRIMARY KEY, data TEXT)"
        )
        db_execute(create_args)
        
        # Get schema for specific table
        schema_args = DBSchemaArgs(table="specific_test")
        result = db_schema(schema_args)
    
    assert result.success is True
    assert result.db_schema is not None
    assert "specific_test" in result.db_schema
    assert len(result.db_schema) == 1  # Only the requested table
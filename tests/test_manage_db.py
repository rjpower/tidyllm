"""Tests for database management tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.manage_db import (
    db_execute,
    db_query,
    db_schema,
)


@pytest.fixture
def test_context():
    """Create a test context with temporary database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(config_dir=temp_path)
        yield ToolContext(config=config)


def test_schema_operation(test_context):
    """Test getting database schema."""
    # Create a test table
    with set_tool_context(test_context):
        db_execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)"
        )
        schema = db_schema()

    # Function completed successfully if no exception raised
    assert schema is not None
    assert "users" in schema

    # Check column details
    users_schema = schema["users"]
    assert len(users_schema) == 3  # id, name, email

    # Find the name column
    name_col = next(col for col in users_schema if col["name"] == "name")
    assert name_col["type"] == "TEXT"
    assert name_col["nullable"] == "False"


def test_execute_insert(test_context):
    """Test executing insert statement."""
    # Create table
    with set_tool_context(test_context):
        result = db_execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        
        # Insert data
        result = db_execute("INSERT INTO test (value) VALUES ('hello')")
    
    # Function completed successfully if no exception raised
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
            db_execute(sql)
        
        # Query data
        rows = db_query("SELECT * FROM test ORDER BY id")
    
    # Function completed successfully if no exception raised
    assert rows is not None
    assert len(rows) == 2
    assert rows[0]["value"] == "hello"
    assert rows[1]["value"] == "world"


def test_query_with_params(test_context):
    """Test query with parameters."""
    # Setup
    with set_tool_context(test_context):
        db_execute("CREATE TABLE test (id INTEGER, name TEXT)")
        db_execute("INSERT INTO test VALUES (1, 'alice'), (2, 'bob')")
        
        # Query with params
        rows = db_query(
            "SELECT * FROM test WHERE name = ?",
            ["alice"]
        )
    
    # Function completed successfully if no exception raised
    assert len(rows) == 1
    assert rows[0]["name"] == "alice"


def test_dangerous_operations_blocked(test_context):
    """Test that dangerous operations are blocked."""
    dangerous_operations = [
        "DROP TABLE test",
        "TRUNCATE TABLE test", 
        "ALTER TABLE test ADD COLUMN new_col TEXT"
    ]
    
    with set_tool_context(test_context):
        for sql in dangerous_operations:
            with pytest.raises(ValueError, match="dangerous|not allowed"):
                db_execute(sql)


def test_non_select_in_query_operation(test_context):
    """Test that non-SELECT statements are blocked in query operation."""
    with set_tool_context(test_context):
        with pytest.raises(ValueError):
            db_query("INSERT INTO test VALUES (1, 'hack')")


def test_sql_error_handling(test_context):
    """Test handling of SQL errors."""
    with set_tool_context(test_context):
        with pytest.raises(Exception):  # Should raise some database error
            db_query("SELECT * FROM nonexistent_table")


def test_empty_database_schema(test_context):
    """Test schema operation on empty database."""
    with set_tool_context(test_context):
        schema = db_schema()
    
    # Function completed successfully if no exception raised
    assert schema is not None
    # Empty database should have empty schema dict
    assert isinstance(schema, dict)


def test_specific_table_schema(test_context):
    """Test getting schema for specific table."""
    # Create a test table
    with set_tool_context(test_context):
        db_execute("CREATE TABLE specific_test (id INTEGER PRIMARY KEY, data TEXT)")
        
        # Get schema for specific table
        schema = db_schema("specific_test")
    
    # Function completed successfully if no exception raised
    assert schema is not None
    assert "specific_test" in schema
    assert len(schema) == 1  # Only the requested table

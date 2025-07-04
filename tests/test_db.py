"""Tests for the Database class and database models."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.database import Database, json_decode, json_encode
from tidyllm.database_models import ColumnSchema, Cursor, Row, Schema, TableSchema


class TestDatabaseModels:
    """Test the database model classes."""

    def test_cursor(self):
        """Test Cursor model."""
        row1 = Row(id=1, name="test1")
        row2 = Row(id=2, name="test2")
        cursor = Cursor(columns=["id", "name"], rows=[row1, row2])

        assert len(cursor) == 2
        assert bool(cursor) is True
        first_row = cursor.first()
        assert first_row is not None
        assert first_row["id"] == 1

        rows = cursor.all()
        assert len(rows) == 2

        # Test iteration
        ids = [row["id"] for row in cursor]
        assert ids == [1, 2]

    def test_empty_cursor(self):
        """Test empty Cursor."""
        cursor = Cursor(columns=[], rows=[])

        assert len(cursor) == 0
        assert bool(cursor) is False
        assert cursor.first() is None
        assert cursor.all() == []


class TestDatabase:
    """Test the Database class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = Database(db_path)
        yield db
        
        db.close()
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def memory_db(self):
        """Create an in-memory database for testing."""
        db = Database(':memory:')
        yield db
        db.close()

    def test_database_init(self, temp_db):
        """Test database initialization."""
        assert temp_db.path.endswith('.db')
        assert temp_db._conn is None

    def test_connect_and_close(self, temp_db):
        """Test database connection and closing."""
        temp_db.connect()
        assert temp_db._conn is not None
        
        temp_db.close()
        assert temp_db._conn is None

    def test_context_manager(self, temp_db):
        """Test database as context manager."""
        with temp_db as db:
            assert db._conn is not None
        assert temp_db._conn is None

    def test_init_schema(self, memory_db):
        """Test schema initialization."""
        memory_db.init_schema()
        
        # Check that vocab table exists
        schema = memory_db.schema()
        vocab_table = schema.get_table("vocab")
        assert vocab_table is not None
        assert len(vocab_table.columns) > 0
        
        # Check specific columns
        column_names = {col.name for col in vocab_table.columns}
        expected_columns = {"id", "word", "translation", "examples", "tags", "created_at", "updated_at"}
        assert expected_columns.issubset(column_names)

    def test_mutate_insert(self, memory_db):
        """Test database mutations (INSERT)."""
        memory_db.init_schema()
        
        # Insert a record
        affected = memory_db.mutate(
            "INSERT INTO vocab (word, translation, examples, tags) VALUES (?, ?, ?, ?)",
            ("hello", "hola", json_encode(["Hello world"]), json_encode(["greetings"]))
        )
        assert affected == 1

    def test_query_select(self, memory_db):
        """Test database queries (SELECT)."""
        memory_db.init_schema()
        
        # Insert test data
        memory_db.mutate(
            "INSERT INTO vocab (word, translation, examples, tags) VALUES (?, ?, ?, ?)",
            ("hello", "hola", json_encode(["Hello world"]), json_encode(["greetings"]))
        )
        
        # Query the data
        cursor = memory_db.query("SELECT * FROM vocab WHERE word = ?", ("hello",))
        
        assert len(cursor) == 1
        row = cursor.first()
        assert row is not None
        assert row["word"] == "hello"
        assert row["translation"] == "hola"
        assert json_decode(row["examples"]) == ["Hello world"]
        assert json_decode(row["tags"]) == ["greetings"]

    def test_query_multiple_rows(self, memory_db):
        """Test querying multiple rows."""
        memory_db.init_schema()
        
        # Insert multiple records
        words = [("hello", "hola"), ("goodbye", "adiós"), ("thanks", "gracias")]
        for word, translation in words:
            memory_db.mutate(
                "INSERT INTO vocab (word, translation) VALUES (?, ?)",
                (word, translation)
            )
        
        # Query all records
        cursor = memory_db.query("SELECT word, translation FROM vocab ORDER BY word")
        
        assert len(cursor) == 3
        results = [(row["word"], row["translation"]) for row in cursor]
        expected = [("goodbye", "adiós"), ("hello", "hola"), ("thanks", "gracias")]
        assert results == expected

    def test_mutate_update(self, memory_db):
        """Test database updates."""
        memory_db.init_schema()
        
        # Insert a record
        memory_db.mutate(
            "INSERT INTO vocab (word, translation) VALUES (?, ?)",
            ("hello", "hola")
        )
        
        # Update the record
        affected = memory_db.mutate(
            "UPDATE vocab SET translation = ? WHERE word = ?",
            ("¡hola!", "hello")
        )
        assert affected == 1
        
        # Verify the update
        cursor = memory_db.query("SELECT translation FROM vocab WHERE word = ?", ("hello",))
        row = cursor.first()
        assert row is not None
        assert row["translation"] == "¡hola!"

    def test_mutate_delete(self, memory_db):
        """Test database deletions."""
        memory_db.init_schema()
        
        # Insert a record
        memory_db.mutate(
            "INSERT INTO vocab (word, translation) VALUES (?, ?)",
            ("hello", "hola")
        )
        
        # Delete the record
        affected = memory_db.mutate("DELETE FROM vocab WHERE word = ?", ("hello",))
        assert affected == 1
        
        # Verify deletion
        cursor = memory_db.query("SELECT * FROM vocab WHERE word = ?", ("hello",))
        assert len(cursor) == 0

    def test_schema_inspection(self, memory_db):
        """Test schema inspection."""
        memory_db.init_schema()
        
        schema = memory_db.schema()
        assert len(schema.tables) >= 1
        
        vocab_table = schema.get_table("vocab")
        assert vocab_table is not None
        assert vocab_table.name == "vocab"
        
        # Check for expected columns
        column_names = [col.name for col in vocab_table.columns]
        assert "id" in column_names
        assert "word" in column_names
        assert "translation" in column_names
        
        # Check primary key
        id_col = next(col for col in vocab_table.columns if col.name == "id")
        assert id_col.primary_key is True
        assert id_col.type == "INTEGER"

    def test_empty_query(self, memory_db):
        """Test querying empty results."""
        memory_db.init_schema()
        
        cursor = memory_db.query("SELECT * FROM vocab WHERE word = ?", ("nonexistent",))
        assert len(cursor) == 0
        assert bool(cursor) is False
        assert cursor.first() is None
        assert cursor.all() == []


class TestUtilityFunctions:
    """Test utility functions."""

    def test_json_encode_decode(self):
        """Test JSON encoding and decoding."""
        # Test with list
        data = ["hello", "world", 123]
        encoded = json_encode(data)
        decoded = json_decode(encoded)
        assert decoded == data
        
        # Test with None
        assert json_encode(None) is None
        assert json_decode(None) == []
        
        # Test with empty string
        assert json_decode("") == []
        
        # Test with invalid JSON
        assert json_decode("invalid json") == []

    def test_json_encode_empty_list(self):
        """Test encoding empty list."""
        encoded = json_encode([])
        assert encoded == "[]"
        decoded = json_decode(encoded)
        assert decoded == []

    def test_json_decode_malformed(self):
        """Test decoding malformed JSON."""
        assert json_decode("{invalid}") == []
        assert json_decode("[1,2,") == []
        assert json_decode("not json at all") == []

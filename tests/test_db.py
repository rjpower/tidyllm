"""Tests for the Database class and database models."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.database import Cursor, Database, Row
from tidyllm.model.serialization import to_json_string, from_json_string


def create_vocab_table(db: Database) -> None:
    """Create the vocab table for testing."""
    db.mutate('''
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL UNIQUE,
            translation TEXT NOT NULL,
            examples TEXT,  -- JSON array
            tags TEXT,      -- JSON array
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create trigger to update updated_at
    db.mutate('''
        CREATE TRIGGER IF NOT EXISTS update_vocab_timestamp 
        AFTER UPDATE ON vocab
        BEGIN
            UPDATE vocab SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
    ''')


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

    def test_context_manager(self, temp_db):
        """Test database as context manager."""
        with temp_db as conn:
            assert conn is not None

    def test_mutate_insert(self, memory_db):
        """Test database mutations (INSERT)."""
        create_vocab_table(memory_db)
        affected = memory_db.mutate(
            "INSERT INTO vocab (word, translation, examples, tags) VALUES (?, ?, ?, ?)",
            ("hello", "hola", to_json_string(["Hello world"]), to_json_string(["greetings"]))
        )
        assert affected == 1

    def test_query_select(self, memory_db):
        """Test database queries (SELECT)."""
        create_vocab_table(memory_db)
        # Insert test data
        memory_db.mutate(
            "INSERT INTO vocab (word, translation, examples, tags) VALUES (?, ?, ?, ?)",
            ("hello", "hola", to_json_string(["Hello world"]), to_json_string(["greetings"]))
        )

        # Query the data
        cursor = memory_db.query("SELECT * FROM vocab WHERE word = ?", ("hello",))

        assert len(cursor) == 1
        row = cursor.first()
        assert row is not None
        assert row["word"] == "hello"
        assert row["translation"] == "hola"
        assert from_json_string(row["examples"], list) == ["Hello world"]
        assert from_json_string(row["tags"], list) == ["greetings"]

    def test_query_multiple_rows(self, memory_db):
        """Test querying multiple rows."""
        create_vocab_table(memory_db)

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
        create_vocab_table(memory_db)

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
        create_vocab_table(memory_db)

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
        create_vocab_table(memory_db)

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
        create_vocab_table(memory_db)

        cursor = memory_db.query("SELECT * FROM vocab WHERE word = ?", ("nonexistent",))
        assert len(cursor) == 0
        assert bool(cursor) is False
        assert cursor.first() is None
        assert cursor.all() == []

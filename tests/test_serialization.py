"""Tests for Pydantic-first serialization module."""

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from tidyllm.linq import Table
from tidyllm.serialization import (
    from_json_dict,
    from_json_dict,
    to_json_dict,
    to_json_string,
    to_json_dict,
)


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Person(BaseModel):
    name: str
    age: int


class TestPydanticSerialization:
    """Test the Pydantic-first serialization functions."""

    def test_none_value(self):
        """Test None values."""
        assert to_json_dict(None) is None
        assert to_json_string(None) == "null"

    def test_primitive_types(self):
        """Test primitive types."""
        assert to_json_dict(42) == 42
        assert to_json_dict(3.14) == 3.14
        assert to_json_dict("hello") == "hello"
        assert to_json_dict(True) is True
        assert to_json_dict(False) is False

    def test_pydantic_models(self):
        """Test Pydantic model serialization."""
        person = Person(name="Alice", age=30)
        result = to_json_dict(person)
        assert result == {"name": "Alice", "age": 30}

    def test_lists(self):
        """Test list serialization."""
        data = [1, "hello", Person(name="Bob", age=25)]
        result = to_json_dict(data)
        expected = [1, "hello", {"name": "Bob", "age": 25}]
        assert result == expected

    def test_dicts(self):
        """Test dictionary serialization."""
        data = {"count": 5, "person": Person(name="Charlie", age=35)}
        result = to_json_dict(data)
        expected = {"count": 5, "person": {"name": "Charlie", "age": 35}}
        assert result == expected

    def test_datetime_types(self):
        """Test datetime serialization with Pydantic."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        d = date(2023, 1, 1)
        t = time(12, 0, 0)
        
        # Pydantic handles these automatically
        assert to_json_dict(dt) == dt.isoformat()
        assert to_json_dict(d) == d.isoformat()
        assert to_json_dict(t) == t.isoformat()

    def test_uuid_path_decimal(self):
        """Test UUID, Path, and Decimal serialization with Pydantic."""
        uuid_val = UUID("550e8400-e29b-41d4-a716-446655440000")
        path_val = Path("/tmp/test")
        decimal_val = Decimal("123.45")
        
        # Pydantic converts these to strings automatically
        assert to_json_dict(uuid_val) == str(uuid_val)
        assert to_json_dict(path_val) == str(path_val)
        assert to_json_dict(decimal_val) == str(decimal_val)

    def test_table_serialization(self):
        """Test Table serialization with Pydantic."""
        people = [
            Person(name="Alice", age=30),
            Person(name="Bob", age=25)
        ]
        table = Table.from_pydantic(people)
        
        # Table is now a Pydantic model, so it serializes directly
        result = to_json_dict(table)
        
        assert "rows" in result
        assert "columns" in result
        assert len(result["rows"]) == 2
        assert result["rows"][0] == {"name": "Alice", "age": 30}
        assert result["rows"][1] == {"name": "Bob", "age": 25}


class TestPydanticDeserialization:
    """Test the Pydantic-first deserialization functions."""

    def test_none_value(self):
        """Test None values."""
        assert from_json_dict(None, str) is None

    def test_primitive_types(self):
        """Test primitive types."""
        assert from_json_dict(42, int) == 42
        assert from_json_dict(3.14, float) == 3.14
        assert from_json_dict("hello", str) == "hello"
        assert from_json_dict(True, bool) is True

    def test_pydantic_models(self):
        """Test Pydantic model deserialization."""
        data = {"name": "Alice", "age": 30}
        result = from_json_dict(data, Person)
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_lists(self):
        """Test list deserialization."""
        data = [1, 2, 3]
        result = from_json_dict(data, list[int])
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)

    def test_dicts(self):
        """Test dictionary deserialization."""
        data = {"a": 1, "b": 2}
        result = from_json_dict(data, dict[str, int])
        assert result == {"a": 1, "b": 2}

    def test_datetime_types(self):
        """Test datetime deserialization with Pydantic."""
        dt_str = "2023-01-01T12:00:00"
        d_str = "2023-01-01"
        t_str = "12:00:00"
        
        dt = from_json_dict(dt_str, datetime)
        d = from_json_dict(d_str, date)
        t = from_json_dict(t_str, time)
        
        assert dt == datetime(2023, 1, 1, 12, 0, 0)
        assert d == date(2023, 1, 1)
        assert t == time(12, 0, 0)

    def test_uuid_path_decimal(self):
        """Test UUID, Path, and Decimal deserialization with Pydantic."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        path_str = "/tmp/test"
        decimal_str = "123.45"
        
        uuid_val = from_json_dict(uuid_str, UUID)
        path_val = from_json_dict(path_str, Path)
        decimal_val = from_json_dict(decimal_str, Decimal)
        
        assert uuid_val == UUID(uuid_str)
        assert path_val == Path(path_str)
        assert decimal_val == Decimal(decimal_str)

    def test_table_deserialization(self):
        """Test Table deserialization with Pydantic."""
        data = {
            "rows": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "columns": {"name": "str", "age": "int"}
        }
        
        result = from_json_dict(data, Table[dict])
        assert isinstance(result, Table)
        assert len(result.rows) == 2
        assert result.rows[0] == {"name": "Alice", "age": 30}
        assert result.rows[1] == {"name": "Bob", "age": 25}
        assert result.columns == {"name": "str", "age": "int"}


class TestRoundTrip:
    """Test round-trip serialization/deserialization with Pydantic."""

    def test_pydantic_model_round_trip(self):
        """Test round-trip for Pydantic models."""
        original = Person(name="Alice", age=30)
        serialized = to_json_dict(original)
        deserialized = from_json_dict(serialized, Person)
        
        assert deserialized == original

    def test_table_round_trip(self):
        """Test round-trip for Table."""
        people = [
            Person(name="Alice", age=30),
            Person(name="Bob", age=25)
        ]
        original = Table.from_pydantic(people)
        
        serialized = to_json_dict(original)
        # Note: Round-trip to Table[dict] since we can't preserve Person type
        deserialized = from_json_dict(serialized, Table[dict])
        
        assert len(deserialized.rows) == len(original.rows)
        assert deserialized.rows[0]["name"] == "Alice"
        assert deserialized.rows[1]["name"] == "Bob"

    def test_complex_data_round_trip(self):
        """Test round-trip for complex nested data."""
        original = {
            "people": [
                Person(name="Alice", age=30),
                Person(name="Bob", age=25)
            ],
            "count": 2,
            "active": True
        }
        
        serialized = to_json_dict(original)
        expected_serialized = {
            "people": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "count": 2,
            "active": True
        }
        assert serialized == expected_serialized

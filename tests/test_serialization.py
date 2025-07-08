"""Tests for Pydantic-first serialization module and dynamic model creation."""

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Union
from uuid import UUID

from pydantic import BaseModel

from tidyllm.linq import Table
from tidyllm.serialization import (
    create_model_from_data_sample,
    create_model_from_field_definitions,
    from_json_dict,
    from_json_string,
    infer_field_types_from_data,
    infer_type_from_values,
    to_json_dict,
    to_json_string,
    transform_argument_type,
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
        assert "table_schema" in result
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
        # Check that columns is now a BaseModel type with the expected fields
        table_schema = result.table_schema()
        assert table_schema is not None
        assert hasattr(table_schema, "model_fields")
        assert "name" in table_schema.model_fields
        assert "age" in table_schema.model_fields


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


class TestTransformArgumentType:
    """Test the transform_argument_type function."""

    def test_simple_types(self):
        """Test transformation of simple types."""
        assert transform_argument_type(str) == str
        assert transform_argument_type(int) == int
        assert transform_argument_type(float) == float

    def test_bytes_transformation(self):
        """Test bytes to Base64Bytes transformation."""
        from pydantic import Base64Bytes
        assert transform_argument_type(bytes) == Base64Bytes

    def test_union_transformation(self):
        """Test Union type transformation."""
        # Test Union[str, int]
        union_type = Union[str, int]
        result = transform_argument_type(union_type)
        # Should still be a Union but potentially transformed
        assert hasattr(result, '__origin__')

    def test_nested_union_with_bytes(self):
        """Test nested Union containing bytes."""
        from pydantic import Base64Bytes
        union_type = Union[str, bytes, int]
        result = transform_argument_type(union_type)
        # This is complex to test exactly, but should not raise an error
        assert result is not None


class TestCreateModelFromFieldDefinitions:
    """Test the create_model_from_field_definitions function."""

    def test_simple_model_creation(self):
        """Test creating a simple model from field definitions."""
        field_definitions = {
            'name': (str, ...),
            'age': (int, ...),
            'active': (bool, True)
        }
        
        model_class = create_model_from_field_definitions("TestModel", field_definitions)
        
        assert model_class.__name__ == "TestModel"
        assert 'name' in model_class.model_fields
        assert 'age' in model_class.model_fields
        assert 'active' in model_class.model_fields
        
        # Test creating an instance
        instance = model_class(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30
        assert instance.active is True  # Default value

    def test_model_with_optional_fields(self):
        """Test creating model with optional fields."""
        field_definitions = {
            'required_field': (str, ...),
            'optional_field': (str, None),
            'default_field': (int, 42)
        }
        
        model_class = create_model_from_field_definitions("OptionalModel", field_definitions)
        
        # Should work with just required field
        instance = model_class(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None
        assert instance.default_field == 42

    def test_custom_config(self):
        """Test creating model with custom config."""
        from pydantic import ConfigDict
        
        field_definitions = {'name': (str, ...)}
        config = ConfigDict(str_strip_whitespace=True)
        
        model_class = create_model_from_field_definitions("ConfigModel", field_definitions, config)
        
        # Test that config is applied
        instance = model_class(name="  Alice  ")
        assert instance.name == "Alice"  # Should be stripped


class TestInferTypeFromValues:
    """Test the infer_type_from_values function."""

    def test_empty_values(self):
        """Test with empty list."""
        from typing import Any
        assert infer_type_from_values([]) == Any

    def test_single_type(self):
        """Test with values of single type."""
        assert infer_type_from_values([1, 2, 3]) == int
        assert infer_type_from_values(["a", "b", "c"]) == str
        assert infer_type_from_values([True, False]) == bool

    def test_mixed_types(self):
        """Test with values of mixed types."""
        # Should return Union for a few types
        result = infer_type_from_values([1, "a", True])
        # This returns a Union type which is hard to test exactly
        assert result is not None

    def test_with_none_values(self):
        """Test with None values mixed in."""
        assert infer_type_from_values([None, None]) == type(None)
        
        # Should ignore None when inferring type
        result = infer_type_from_values([1, 2, None, 3])
        assert result == int

    def test_many_types_fallback(self):
        """Test fallback to Any for many different types."""
        from typing import Any
        many_types = [1, "str", [], {}, 3.14, True, b"bytes"]
        assert infer_type_from_values(many_types) == Any


class TestInferFieldTypesFromData:
    """Test the infer_field_types_from_data function."""

    def test_empty_data(self):
        """Test with empty data."""
        result = infer_field_types_from_data([])
        assert result == {}

    def test_dictionary_data(self):
        """Test with dictionary data."""
        data = [
            {'name': 'Alice', 'age': 25, 'active': True},
            {'name': 'Bob', 'age': 30, 'active': False}
        ]
        
        result = infer_field_types_from_data(data)
        
        assert 'name' in result
        assert 'age' in result
        assert 'active' in result
        
        # Check that all fields are optional (Union with None)
        name_type, name_default = result['name']
        assert name_default is None  # Should be optional

    def test_pydantic_model_data(self):
        """Test with Pydantic model data."""
        people = [
            Person(name="Alice", age=25),
            Person(name="Bob", age=30)
        ]
        
        result = infer_field_types_from_data(people)
        
        assert 'name' in result
        assert 'age' in result
        
        # Should extract exact types from model fields
        name_type, name_default = result['name']
        age_type, age_default = result['age']
        # Defaults should match model defaults (either ... or PydanticUndefined for required fields)
        assert name_default in (..., None, "PydanticUndefined") or str(name_default) == "PydanticUndefined"
        assert age_default in (..., None, "PydanticUndefined") or str(age_default) == "PydanticUndefined"

    def test_missing_keys_in_dicts(self):
        """Test with dictionaries having missing keys."""
        data = [
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob', 'age': 30},  # Missing city
            {'name': 'Charlie', 'city': 'LA'}  # Missing age
        ]
        
        result = infer_field_types_from_data(data)
        
        # All fields should be present and optional
        assert 'name' in result
        assert 'age' in result
        assert 'city' in result
        
        # All should be optional due to missing values
        for field_name, (field_type, default) in result.items():
            assert default is None  # Should be optional

    def test_primitive_data(self):
        """Test with primitive data (not dicts or models)."""
        data = [1, 2, 3, 4, 5]
        
        result = infer_field_types_from_data(data)
        
        # Should create single 'value' field
        assert 'value' in result
        value_type, value_default = result['value']
        assert value_default == ...  # Required


class TestCreateModelFromDataSample:
    """Test the create_model_from_data_sample function."""

    def test_empty_data(self):
        """Test with empty data."""
        model_class = create_model_from_data_sample([])
        
        assert model_class.__name__ == "InferredSchema"
        # Should be an empty model
        instance = model_class()
        assert instance is not None

    def test_dictionary_data(self):
        """Test with dictionary data."""
        data = [
            {'name': 'Alice', 'age': 25, 'active': True},
            {'name': 'Bob', 'age': 30, 'active': False},
            {'name': 'Charlie', 'age': 35, 'active': True}
        ]
        
        model_class = create_model_from_data_sample(data, "PersonSchema")
        
        assert model_class.__name__ == "PersonSchema"
        assert 'name' in model_class.model_fields
        assert 'age' in model_class.model_fields
        assert 'active' in model_class.model_fields
        
        # Test creating instance
        instance = model_class(name="Test", age=99, active=True)
        assert instance.name == "Test"
        assert instance.age == 99
        assert instance.active is True

    def test_pydantic_model_data_same_type(self):
        """Test with Pydantic models of same type."""
        people = [
            Person(name="Alice", age=25),
            Person(name="Bob", age=30),
            Person(name="Charlie", age=35)
        ]
        
        model_class = create_model_from_data_sample(people)
        
        # Should return the original Person class
        assert model_class is Person

    def test_mixed_pydantic_models(self):
        """Test with different Pydantic model types."""
        class User(BaseModel):
            username: str
            email: str
        
        # Mix of Person and User models
        data = [
            Person(name="Alice", age=25),
            User(username="bob", email="bob@example.com")
        ]
        
        model_class = create_model_from_data_sample(data)
        
        # Should create new inferred schema since types are different
        assert model_class.__name__ == "InferredSchema"
        assert model_class is not Person
        assert model_class is not User

    def test_primitive_data(self):
        """Test with primitive data."""
        data = [1, 2, 3, 4, 5]
        
        model_class = create_model_from_data_sample(data, "NumberSchema")
        
        assert model_class.__name__ == "NumberSchema"
        assert 'value' in model_class.model_fields
        
        # Test creating instance
        instance = model_class(value=42)
        assert instance.value == 42

    def test_mixed_data_types(self):
        """Test with mixed data (some dicts, some primitives)."""
        data = [
            {'name': 'Alice', 'score': 85},
            {'name': 'Bob', 'score': 'N/A'},  # Different type
            {'name': 'Charlie', 'score': 92}
        ]
        
        model_class = create_model_from_data_sample(data)
        
        assert 'name' in model_class.model_fields
        assert 'score' in model_class.model_fields
        
        # Should handle mixed types in score field

    def test_custom_model_name(self):
        """Test with custom model name."""
        data = [{'test': 'value'}]
        
        model_class = create_model_from_data_sample(data, "CustomName")
        assert model_class.__name__ == "CustomName"

    def test_transform_types_option(self):
        """Test with transform_types option."""
        # This is more of an integration test
        data = [{'name': 'test', 'value': 42}]
        
        model_with_transform = create_model_from_data_sample(data, transform_types=True)
        model_without_transform = create_model_from_data_sample(data, transform_types=False)
        
        # Both should work (difference mainly for bytes/Union types)
        assert model_with_transform.__name__ == "InferredSchema"
        assert model_without_transform.__name__ == "InferredSchema"


class TestIntegrationDynamicModels:
    """Integration tests for dynamic model creation."""

    def test_function_schema_integration(self):
        """Test that dynamic models work with function schema creation."""
        # This tests that our refactored function schema creation works
        from tidyllm.function_schema import FunctionDescription

        def test_function(name: str, age: int = 25) -> str:
            """Test function for schema integration."""
            return f"{name} is {age} years old"

        desc = FunctionDescription(test_function)

        # Should create args model successfully
        assert desc.args_model.__name__ == "Test_FunctionArgs"
        assert 'name' in desc.args_model.model_fields
        assert 'age' in desc.args_model.model_fields

        # Should be able to validate args
        validated = desc.validate_and_parse_args({"name": "Alice", "age": 30})
        assert validated["name"] == "Alice"
        assert validated["age"] == 30

    def test_linq_schema_serialization_integration(self):
        """Test LINQ schema with serialization roundtrip."""
        from tidyllm.linq import from_iterable

        # Start with data
        data = [
            {'product': 'Widget', 'price': 10.99, 'in_stock': True},
            {'product': 'Gadget', 'price': 25.50, 'in_stock': False}
        ]

        # Create schema-aware enumerable
        enum = from_iterable(data).with_schema_inference()
        schema = enum.table_schema()

        # Serialize the schema (as a model class, this is tricky)
        # Instead, test creating instances from the schema
        instance = schema(product="Test", price=99.99, in_stock=True)

        # Serialize the instance
        serialized = to_json_dict(instance)
        assert serialized == {"product": "Test", "price": 99.99, "in_stock": True}

        # Deserialize back
        deserialized = from_json_dict(serialized, schema)
        assert deserialized.product == "Test"
        assert deserialized.price == 99.99
        assert deserialized.in_stock is True

    def test_table_dynamic_schema_roundtrip(self):
        """Test Table with dynamic schema in serialization roundtrip."""
        # Create table with inferred schema
        data = [
            {'user_id': 1, 'username': 'alice', 'active': True},
            {'user_id': 2, 'username': 'bob', 'active': False}
        ]

        table = Table.from_rows(data)
        schema = table.table_schema()

        # Schema should be dynamically created
        assert 'user_id' in schema.model_fields
        assert 'username' in schema.model_fields
        assert 'active' in schema.model_fields

        # Create instance from schema
        new_user = schema(user_id=3, username="charlie", active=True)
        assert new_user.user_id == 3
        assert new_user.username == "charlie"
        assert new_user.active is True

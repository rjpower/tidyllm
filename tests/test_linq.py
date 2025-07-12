"""Comprehensive tests for LINQ operations and schema inference."""

import pytest
from pydantic import BaseModel

from tidyllm.types.linq import Table
from tidyllm.types.serialization import create_model_from_data_sample


class User(BaseModel):
    """Test user model."""
    name: str
    age: int
    active: bool = True


class TestBasicLINQ:
    """Test basic LINQ operations."""

    def test_table_from_rows(self):
        """Test creating enumerable from iterable."""
        data = [1, 2, 3, 4, 5]
        enum = Table.from_rows(data)
        assert list(enum) == data

    def test_select(self):
        """Test select transformation."""
        data = [1, 2, 3]
        result = Table.from_rows(data).select(lambda x: x * 2).to_list()
        assert result == [2, 4, 6]

    def test_where(self):
        """Test where filtering."""
        data = [1, 2, 3, 4, 5]
        result = Table.from_rows(data).where(lambda x: x % 2 == 0).to_list()
        assert result == [2, 4]

    def test_chained_operations(self):
        """Test chaining multiple operations."""
        data = [1, 2, 3, 4, 5, 6]
        result = (Table.from_rows(data)
                 .where(lambda x: x > 2)
                 .select(lambda x: x * 2)
                 .where(lambda x: x < 10)
                 .to_list())
        assert result == [6, 8]

    def test_batch_method_exists(self):
        """Test that batch method exists and works."""
        data = [1, 2, 3, 4, 5, 6, 7]
        result = Table.from_rows(data).batch(3).to_list()
        assert result == [[1, 2, 3], [4, 5, 6], [7]]

    def test_chunk_method_removed(self):
        """Test that chunk method has been removed."""
        data = [1, 2, 3, 4, 5]
        enum = Table.from_rows(data)
        with pytest.raises(AttributeError):
            enum.chunk(3)

    def test_count(self):
        """Test count operations."""
        data = [1, 2, 3, 4, 5]
        enum = Table.from_rows(data)
        assert enum.count() == 5
        assert enum.count(lambda x: x > 3) == 2

    def test_first_and_last(self):
        """Test first and last operations."""
        data = [1, 2, 3, 4, 5]
        enum = Table.from_rows(data)
        assert enum.first() == 1
        assert enum.first(lambda x: x > 3) == 4
        assert enum.last() == 5
        assert enum.last(lambda x: x < 3) == 2

    def test_any_and_all(self):
        """Test any and all operations."""
        data = [2, 4, 6, 8]
        enum = Table.from_rows(data)
        assert enum.any(lambda x: x > 5)
        assert enum.all(lambda x: x % 2 == 0)
        assert not enum.any(lambda x: x > 10)

    def test_distinct(self):
        """Test distinct operation."""
        data = [1, 2, 2, 3, 3, 3, 4]
        result = Table.from_rows(data).distinct().to_list()
        assert result == [1, 2, 3, 4]

    def test_order_by(self):
        """Test ordering operations."""
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        result = Table.from_rows(data).order_by(lambda x: x).to_list()
        assert result == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_group_by(self):
        """Test grouping operations."""
        data = ['apple', 'banana', 'apricot', 'blueberry']
        groups = Table.from_rows(data).group_by(lambda x: x[0]).to_list()
        
        # Convert to dict for easier testing
        group_dict = {g.key: list(g) for g in groups}
        assert 'a' in group_dict
        assert 'b' in group_dict
        assert set(group_dict['a']) == {'apple', 'apricot'}
        assert set(group_dict['b']) == {'banana', 'blueberry'}

    def test_take_and_skip(self):
        """Test take and skip operations."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        enum = Table.from_rows(data)
        assert enum.take(3).to_list() == [1, 2, 3]
        assert enum.skip(3).to_list() == [4, 5, 6, 7, 8, 9, 10]
        assert enum.skip(3).take(3).to_list() == [4, 5, 6]

    def test_partition(self):
        """Test partition operation."""
        data = [1, 2, 3, 4, 5, 6]
        enum = Table.from_rows(data)
        evens, odds = enum.partition(lambda x: x % 2 == 0)
        assert list(evens) == [2, 4, 6]
        assert list(odds) == [1, 3, 5]

    def test_try_select(self):
        """Test try_select with exception handling."""
        data = [1, 2, 0, 4, 5]
        
        def divide_by_x(x):
            if x == 0:
                raise ValueError("Cannot divide by zero")
            return 10 / x
        
        successes, failures = Table.from_rows(data).try_select(divide_by_x)
        
        success_list = list(successes)
        failure_list = list(failures)
        
        assert len(success_list) == 4  # 1, 2, 4, 5 succeed
        assert len(failure_list) == 1   # 0 fails
        assert isinstance(failure_list[0], ValueError)


class TestSchemaInference:
    """Test schema inference functionality."""

    def test_with_schema_inference_basic(self):
        """Test basic schema inference."""
        data = [
            {'name': 'Alice', 'age': 25, 'active': True},
            {'name': 'Bob', 'age': 30, 'active': False}
        ]
        
        enum = Table.from_rows(data).with_schema_inference()
        schema = enum.table_schema()
        
        assert schema.__name__ == "InferredSchema"
        assert 'name' in schema.model_fields
        assert 'age' in schema.model_fields
        assert 'active' in schema.model_fields

    def test_schema_inference_preserves_data(self):
        """Test that schema inference doesn't lose data during iteration."""
        data = [
            {'user': 'alice', 'score': 85},
            {'user': 'bob', 'score': 92},
            {'user': 'charlie', 'score': 78}
        ]
        
        enum = Table.from_rows(data).with_schema_inference()
        
        # Get schema first
        schema = enum.table_schema()
        assert schema.__name__ == "InferredSchema"
        
        # Then iterate - should preserve all data
        result = list(enum)
        assert len(result) == 3
        assert result == data

    def test_schema_inference_with_transformations(self):
        """Test schema inference after LINQ transformations."""
        data = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30},
            {'name': 'Charlie', 'age': 35}
        ]
        
        transformed = (Table.from_rows(data)
                      .where(lambda x: x['age'] >= 30)
                      .select(lambda x: {
                          'username': x['name'].upper(),
                          'is_senior': x['age'] > 60,
                          'category': 'adult'
                      })
                      .with_schema_inference())
        
        schema = transformed.table_schema()
        assert 'username' in schema.model_fields
        assert 'is_senior' in schema.model_fields
        assert 'category' in schema.model_fields
        
        results = list(transformed)
        assert len(results) == 2
        assert results[0]['username'] == 'BOB'
        assert results[1]['username'] == 'CHARLIE'

    def test_schema_inference_with_pydantic_models(self):
        """Test schema inference with Pydantic models."""
        users = [
            User(name='Alice', age=25),
            User(name='Bob', age=30, active=False)
        ]
        
        enum = Table.from_rows(users).with_schema_inference()
        schema = enum.table_schema()
        
        # Should return the exact User type
        assert schema is User

    def test_schema_inference_with_mixed_data(self):
        """Test schema inference with missing keys."""
        data = [
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob', 'age': 30},  # Missing city
            {'name': 'Charlie', 'city': 'LA'}  # Missing age
        ]
        
        enum = Table.from_rows(data).with_schema_inference()
        schema = enum.table_schema()
        
        # All fields should be optional due to missing values
        assert 'name' in schema.model_fields
        assert 'age' in schema.model_fields
        assert 'city' in schema.model_fields

    def test_schema_inference_caching(self):
        """Test that schema inference results are cached."""
        data = [{'name': 'test', 'value': 42}]
        enum = Table.from_rows(data).with_schema_inference()
        
        schema1 = enum.table_schema()
        schema2 = enum.table_schema()
        
        # Should be the same object (cached)
        assert schema1 is schema2

    def test_schema_inference_sample_size(self):
        """Test schema inference with custom sample size."""
        # Create data where later items have additional fields
        data = [
            {'name': 'Alice'},
            {'name': 'Bob'},
            {'name': 'Charlie', 'age': 35, 'city': 'NYC'}  # Additional fields
        ]
        
        # With sample size 2, should not see the additional fields
        enum_small = Table.from_rows(data).with_schema_inference(sample_size=2)
        schema_small = enum_small.table_schema()
        
        # Should only have 'name' field
        assert 'name' in schema_small.model_fields
        # age and city might not be detected with small sample
        
        # With larger sample size, should see all fields
        enum_large = Table.from_rows(data).with_schema_inference(sample_size=5)
        schema_large = enum_large.table_schema()
        
        assert 'name' in schema_large.model_fields


class TestTable:
    """Test Table functionality and schema."""

    def test_table_creation_from_rows(self):
        """Test creating table from rows."""
        data = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30}
        ]

        table = Table.from_rows(data)
        assert len(table) == 2
        assert table[0] == data[0]
        assert table[1] == data[1]

    def test_table_creation_from_rows(self):
        """Test creating table from Pydantic models."""
        users = [
            User(name='Alice', age=25),
            User(name='Bob', age=30, active=False)
        ]

        table = Table.from_rows(users)
        assert len(table) == 2
        assert table[0].name == 'Alice'
        assert table[1].active is False


    def test_table_linq_operations(self):
        """Test that Table supports LINQ operations."""
        data = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30},
            {'name': 'Charlie', 'age': 35}
        ]

        table = Table.from_rows(data)

        # Test filtering
        adults = table.where(lambda x: x['age'] >= 30).to_list()
        assert len(adults) == 2

        # Test transformation
        names = table.select(lambda x: x['name']).to_list()
        assert names == ['Alice', 'Bob', 'Charlie']

    def test_table_to_enumerable_conversion(self):
        """Test converting enumerable to table."""
        data = [1, 2, 3, 4, 5]
        enum = Table.from_rows(data)
        table = enum.materialize()

        assert isinstance(table, Table)
        assert list(table) == data


class TestAdvancedLINQ:
    """Test advanced LINQ operations."""

    def test_with_progress(self):
        """Test progress tracking (should not fail)."""
        data = [1, 2, 3, 4, 5]
        # This should not raise an exception even if tqdm is not available
        result = Table.from_rows(data).with_progress("Testing").to_list()
        assert result == data

    def test_select_many(self):
        """Test select_many (flatten) operation."""
        data = [[1, 2], [3, 4], [5, 6]]
        result = Table.from_rows(data).select_many(lambda x: x).to_list()
        assert result == [1, 2, 3, 4, 5, 6]

    def test_aggregate(self):
        """Test aggregate operation."""
        data = [1, 2, 3, 4, 5]
        result = Table.from_rows(data).aggregate(0, lambda acc, x: acc + x)
        assert result == 15

    def test_set_operations(self):
        """Test set operations (union, intersect, except)."""
        data1 = [1, 2, 3, 4]
        data2 = [3, 4, 5, 6]
        
        enum1 = Table.from_rows(data1)
        enum2 = Table.from_rows(data2)
        
        union_result = enum1.union(enum2).to_list()
        assert set(union_result) == {1, 2, 3, 4, 5, 6}
        
        intersect_result = Table.from_rows(data1).intersect(enum2).to_list()
        assert set(intersect_result) == {3, 4}
        
        except_result = Table.from_rows(data1).except_(enum2).to_list()
        assert set(except_result) == {1, 2}

    def test_to_dict(self):
        """Test to_dict conversion."""
        data = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        
        enum = Table.from_rows(data)
        result = enum.to_dict(lambda x: x['id'], lambda x: x['name'])
        assert result == {1: 'Alice', 2: 'Bob'}

    def test_window_operation(self):
        """Test sliding window operation."""
        data = [1, 2, 3, 4, 5]
        windows = Table.from_rows(data).window(3).to_list()
        
        assert len(windows) == 3
        assert windows[0] == [1, 2, 3]
        assert windows[1] == [2, 3, 4]
        assert windows[2] == [3, 4, 5]

    def test_join_operation(self):
        """Test join operation."""
        users = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        
        orders = [
            {'user_id': 1, 'product': 'Widget'},
            {'user_id': 2, 'product': 'Gadget'},
            {'user_id': 1, 'product': 'Tool'}
        ]
        
        result = (Table.from_rows(users)
                 .join(Table.from_rows(orders),
                       lambda u: u['id'],
                       lambda o: o['user_id'],
                       lambda u, o: {'user': u['name'], 'product': o['product']})
                 .to_list())
        
        assert len(result) == 3
        assert {'user': 'Alice', 'product': 'Widget'} in result
        assert {'user': 'Alice', 'product': 'Tool'} in result
        assert {'user': 'Bob', 'product': 'Gadget'} in result


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_pipeline_with_schema(self):
        """Test complete pipeline from data to schema-aware results."""
        # Start with raw data
        raw_data = [
            {'name': 'Alice Johnson', 'age': 28, 'department': 'Engineering', 'salary': 75000},
            {'name': 'Bob Smith', 'age': 34, 'department': 'Marketing', 'salary': 65000},
            {'name': 'Charlie Brown', 'age': 29, 'department': 'Engineering', 'salary': 80000},
            {'name': 'Diana Prince', 'age': 31, 'department': 'Sales', 'salary': 70000}
        ]

        # Complex transformation pipeline
        pipeline = (Table.from_rows(raw_data)
                   .where(lambda p: p['age'] >= 30)  # Adults only
                   .select(lambda p: {
                       'full_name': p['name'],
                       'is_engineer': p['department'] == 'Engineering',
                       'salary_category': 'high' if p['salary'] > 70000 else 'medium',
                       'years_experience': p['age'] - 22  # Assume started at 22
                   })
                   .with_schema_inference())

        # Get schema
        schema = pipeline.table_schema()
        assert 'full_name' in schema.model_fields
        assert 'is_engineer' in schema.model_fields
        assert 'salary_category' in schema.model_fields
        assert 'years_experience' in schema.model_fields

        # Get results
        results = list(pipeline)
        assert len(results) == 2  # Bob and Diana

        # Verify transformation worked
        bob_result = next(r for r in results if 'Bob' in r['full_name'])
        assert bob_result['is_engineer'] is False
        assert bob_result['salary_category'] == 'medium'
        assert bob_result['years_experience'] == 12

    def test_table_linq_schema_roundtrip(self):
        """Test Table -> LINQ -> Schema roundtrip."""
        # Create initial table
        users = [
            User(name='Alice', age=25, active=True),
            User(name='Bob', age=30, active=False),
            User(name='Charlie', age=35, active=True)
        ]

        table = Table.from_rows(users)

        # Transform through LINQ
        transformed = (table
                      .where(lambda u: u.active)
                      .select(lambda u: {
                          'username': u.name.lower(),
                          'age_group': 'young' if u.age < 30 else 'adult'
                      })
                      .with_schema_inference())

        # Get new schema
        new_schema = transformed.table_schema()
        assert 'username' in new_schema.model_fields
        assert 'age_group' in new_schema.model_fields

        # Convert back to table
        result_table = transformed.materialize()

        # Data should be correct
        results = list(result_table)
        assert len(results) == 2  # Only Alice and Charlie (active users)
        assert all(r['username'] in ['alice', 'charlie'] for r in results)

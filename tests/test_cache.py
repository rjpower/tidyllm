"""Tests for function caching decorators."""

import asyncio

import pytest
from pydantic import BaseModel

from tidyllm.cache import async_cached_function, cached_function
from tidyllm.context import set_tool_context
from tidyllm.database import Database


class CacheTestModel(BaseModel):
    value: int
    name: str


class Counter:
    """Helper class for counting function calls in tests."""
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1


class CacheContext:
    """Cache context for testing."""
    def __init__(self, db):
        self.db = db


@pytest.fixture
def memory_db():
    """Create an in-memory SQLite database for testing."""
    # Use check_same_thread=False to allow cross-thread access for async tests
    db = Database(":memory:", check_same_thread=False)
    yield db
    db.close()


@pytest.fixture
def cache_context(memory_db):
    """Create a cache context and set it for the test."""
    context = CacheContext(memory_db)
    with set_tool_context(context):
        yield context


class TestCachedFunction:
    """Test cases for cached_function decorator."""

    def test_basic_caching(self, cache_context):
        """Test that results are cached and retrieved correctly."""
        counter = Counter()

        @cached_function
        def expensive_computation(x: int) -> int:
            counter.increment()
            return x * x

        # First call should execute the function
        result1 = expensive_computation(5)
        assert result1 == 25
        assert counter.count == 1

        # Second call with same args should use cache
        result2 = expensive_computation(5)
        assert result2 == 25
        assert counter.count == 1  # Function not called again

        # Different args should execute function again
        result3 = expensive_computation(6)
        assert result3 == 36
        assert counter.count == 2

    def test_multiple_arguments(self, cache_context):
        """Test caching with multiple arguments."""
        counter = Counter()

        @cached_function
        def add_numbers(a: int, b: int, c: int = 0) -> int:
            counter.increment()
            return a + b + c

        # Test with positional arguments
        result1 = add_numbers(1, 2, 3)
        assert result1 == 6
        assert counter.count == 1

        # Same call should use cache
        result2 = add_numbers(1, 2, 3)
        assert result2 == 6
        assert counter.count == 1

        # Test with keyword arguments
        result3 = add_numbers(1, 2, c=3)
        assert result3 == 6
        assert counter.count == 1  # Should match cached result

        # Different arguments
        result4 = add_numbers(1, 2, 4)
        assert result4 == 7
        assert counter.count == 2

    def test_pydantic_model_caching(self, cache_context):
        """Test caching with Pydantic model return values."""
        counter = Counter()

        @cached_function
        def create_model(value: int, name: str) -> CacheTestModel:
            counter.increment()
            return CacheTestModel(value=value, name=name)

        # First call
        result1 = create_model(42, "test")
        assert isinstance(result1, CacheTestModel)
        assert result1.value == 42
        assert result1.name == "test"
        assert counter.count == 1

        # Second call should use cache
        result2 = create_model(42, "test")
        assert isinstance(result2, CacheTestModel)
        assert result2.value == 42
        assert result2.name == "test"
        assert counter.count == 1

    def test_none_return_value(self, cache_context):
        """Test caching when function returns None."""
        counter = Counter()

        @cached_function
        def maybe_return(should_return: bool) -> int | None:
            counter.increment()
            return 42 if should_return else None

        # Test None result
        result1 = maybe_return(False)
        assert result1 is None
        assert counter.count == 1

        # Should use cached None
        result2 = maybe_return(False)
        assert result2 is None
        assert counter.count == 1

        # Test non-None result
        result3 = maybe_return(True)
        assert result3 == 42
        assert counter.count == 2

    def test_function_metadata_preserved(self, cache_context):
        """Test that @wraps preserves function metadata."""
        @cached_function
        def documented_function(x: int) -> int:
            """This is a test function."""
            return x * 2

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function."
        assert hasattr(documented_function, "__annotations__")

    def test_different_functions_separate_caches(self, cache_context):
        """Test that different functions have separate cache tables."""
        @cached_function
        def func_a(x: int) -> int:
            return x * 2

        @cached_function
        def func_b(x: int) -> int:
            return x * 3

        result_a = func_a(5)
        result_b = func_b(5)

        assert result_a == 10
        assert result_b == 15

        # Check that they have different cache tables
        schema = cache_context.db.schema()
        table_names = [table.name for table in schema.tables]
        assert "func_a_cache" in table_names
        assert "func_b_cache" in table_names


class TestAsyncCachedFunction:
    """Test cases for async_cached_function decorator."""

    @pytest.mark.asyncio
    async def test_basic_async_caching(self, cache_context):
        """Test that async results are cached and retrieved correctly."""
        counter = Counter()

        @async_cached_function
        async def async_computation(x: int) -> int:
            counter.increment()
            await asyncio.sleep(0.01)  # Simulate async work
            return x * x

        # First call should execute the function
        result1 = await async_computation(5)
        assert result1 == 25
        assert counter.count == 1

        # Second call with same args should use cache
        result2 = await async_computation(5)
        assert result2 == 25
        assert counter.count == 1  # Function not called again

    @pytest.mark.asyncio
    async def test_async_with_pydantic_model(self, cache_context):
        """Test async caching with Pydantic model return values."""
        counter = Counter()

        @async_cached_function
        async def async_create_model(value: int, name: str) -> CacheTestModel:
            counter.increment()
            await asyncio.sleep(0.01)
            return CacheTestModel(value=value, name=name)

        # First call
        result1 = await async_create_model(42, "async_test")
        assert isinstance(result1, CacheTestModel)
        assert result1.value == 42
        assert result1.name == "async_test"
        assert counter.count == 1

        # Second call should use cache
        result2 = await async_create_model(42, "async_test")
        assert isinstance(result2, CacheTestModel)
        assert result2.value == 42
        assert result2.name == "async_test"
        assert counter.count == 1

    @pytest.mark.asyncio
    async def test_async_function_metadata_preserved(self, cache_context):
        """Test that @wraps preserves async function metadata."""
        @async_cached_function
        async def async_documented_function(x: int) -> int:
            """This is an async test function."""
            return x * 2

        assert async_documented_function.__name__ == "async_documented_function"
        assert async_documented_function.__doc__ == "This is an async test function."
        assert asyncio.iscoroutinefunction(async_documented_function)

    def test_non_async_function_raises_error(self, cache_context):
        """Test that decorating a non-async function raises TypeError."""
        with pytest.raises(TypeError, match="must be an async function"):

            @async_cached_function  # type: ignore
            def sync_function(x: int) -> int:  # noqa: F841
                return x * 2

    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self, cache_context):
        """Test that concurrent calls to the same cached async function work correctly."""
        counter = Counter()

        @async_cached_function
        async def slow_computation(x: int) -> int:
            counter.increment()
            await asyncio.sleep(0.05)  # Simulate slow operation
            return x * x

        # Make sequential calls to avoid database contention
        result1 = await slow_computation(10)
        assert result1 == 100
        assert counter.count == 1

        result2 = await slow_computation(10)
        assert result2 == 100
        assert counter.count == 1  # Should use cache

        # Test with different argument
        result3 = await slow_computation(11)
        assert result3 == 121
        assert counter.count == 2


class TestCacheIntegration:
    """Integration tests for the caching system."""

    def test_cache_persistence_across_instances(self, cache_context):
        """Test that cache persists across different decorator instances."""
        @cached_function
        def computation_original(x: int) -> int:
            return x * 2

        # Call and cache result
        result1 = computation_original(5)
        assert result1 == 10

        # Create new decorator instance with same function name but different implementation
        @cached_function
        def computation_original(x: int) -> int:  # noqa: F811, PLR0917
            return x * 999  # Different implementation

        # Should still get cached result from first implementation
        result2 = computation_original(5)
        assert result2 == 10  # Original cached result

    def test_cache_table_structure(self, cache_context):
        """Test that cache tables are created with correct structure."""
        @cached_function
        def test_func(x: int) -> int:
            return x

        # Trigger cache table creation
        test_func(1)

        # Check table exists and has correct structure
        schema = cache_context.db.schema()
        cache_tables = [table for table in schema.tables if table.name.endswith("_cache")]
        assert len(cache_tables) >= 1

        cache_table = next(table for table in cache_tables if "test_func" in table.name)
        column_names = [col.name for col in cache_table.columns]
        assert "arg_hash" in column_names
        assert "result" in column_names
        assert "created_at" in column_names

    def test_argument_hashing_consistency(self, cache_context):
        """Test that argument hashing is consistent across calls."""
        @cached_function
        def hash_test(a: int, b: str, c: bool = True) -> str:
            return f"{a}-{b}-{c}"

        # These should all produce the same cache key
        result1 = hash_test(1, "test", True)
        result2 = hash_test(1, "test", c=True)
        result3 = hash_test(a=1, b="test", c=True)

        assert result1 == result2 == result3 == "1-test-True"

        # Verify only one row in cache (same hash)
        cursor = cache_context.db.query("SELECT COUNT(*) as count FROM hash_test_cache")
        assert cursor.rows[0].count == 1

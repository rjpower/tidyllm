# TidyLLM Code Refactoring Specification: LINQ-Style Enumerables

## Overview

This document specifies the refactoring of `tidyllm/data.py` and `tidyllm/schema.py` into three focused modules:

1. **`serialization.py`** - JSON serialization helpers and type registration
2. **`function_schema.py`** - Function description and schema generation for tool calling
3. **`linq.py`** - LINQ-style enumerable implementation with deferred evaluation

## Current State Analysis

### data.py (323 lines)
- **Serialization functions**: `parse_from_json()`, `to_json_value()` 
- **Type definitions**: `Serializable` type alias, `ColumnSchema` type alias
- **Table classes**: `Table` (protocol/interface), `ConcreteTable` (implementation)
- **Basic operations**: `map()`, `filter()` on ConcreteTable
- **Sequence protocol**

### schema.py (296 lines)
- **Function management**: `FunctionDescription` class
- **Schema generation**: JSON schema creation for OpenAI-compatible tools
- **Type registration**: Registry for custom type transformations
- **Async handling**: Support for async function calls
- **Validation**: Pydantic model creation and validation

## Module 1: serialization.py

### Purpose
Centralize all JSON serialization/deserialization logic and provide a type registration system.

### Contents
```python
# Type definitions
Serializable: TypeAlias = Union[
    int, float, str, bytes, Enum, date, datetime, UUID, Path, Decimal,
    BaseModel, "dict[str, Serializable]", "list[Serializable]", "Enumerable"
]

# Core functions
def parse_from_json(value: Serializable, value_type: type) -> Any
def to_json_value(value: Serializable) -> Any

# Type registration system
class SerializationRegistry:
    _serializers: dict[type, Callable[[Any], Any]]
    _deserializers: dict[type, Callable[[Any], Any]]
    
    @classmethod
    def register_serializer(cls, type_: type, serializer: Callable)
    
    @classmethod
    def register_deserializer(cls, type_: type, deserializer: Callable)
    
    @classmethod
    def serialize(cls, value: Any) -> Any
    
    @classmethod
    def deserialize(cls, value: Any, target_type: type) -> Any

# Default registrations
SerializationRegistry.register_serializer(bytes, lambda b: base64.b64encode(b).decode())
SerializationRegistry.register_deserializer(bytes, lambda s: base64.b64decode(s))
# ... other default types
```

### Dependencies
- Standard library types
- Pydantic BaseModel
- No circular dependencies

## Module 2: function_schema.py

### Purpose
Handle function wrapping, schema generation, and validation for tool calling.

### Contents
```python
from tidyllm.serialization import SerializationRegistry

# Type definitions
class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]

class JSONSchema(TypedDict):
    type: str
    function: FunctionSchema

# Core classes
class FunctionDescription:
    """Enhanced with serialization registry integration"""
    
    function: Callable
    function_schema: JSONSchema
    name: str
    description: str
    tags: list[str]
    result_type: type
    args_model: type[BaseModel]
    args_json_schema: dict
    is_async: bool
    
    def __init__(self, func: Callable, name: str | None = None, 
                 doc_override: str | None = None, description: str = "", 
                 tags: list[str] | None = None)
    
    def validate_and_parse_args(self, json_args: dict) -> dict
    def call(self, *args, **kwargs) -> Any
    async def call_async(self, *args, **kwargs) -> Any
    
    # Type registration moved to SerializationRegistry
    # Remove _registry class variable

# Helper functions
def function_schema_from_args(args_json_schema: dict, name: str, doc: str) -> JSONSchema
def _process_union_type(param_type: Any) -> Any  # Now uses SerializationRegistry
```

### Dependencies
- `tidyllm.serialization` for type handling
- `tidyllm.docstring` for documentation extraction
- Pydantic for model creation

## Module 3: linq.py

### Purpose
Provide a LINQ-style enumerable implementation with deferred evaluation, strong typing, and comprehensive query operations.

### Design Principles
1. **Deferred Evaluation**: Build operation graphs that execute lazily
2. **Type Safety**: Full type annotations and generic support
3. **Composability**: Chain operations fluently
4. **Performance**: Minimize intermediate materializations

### Core Architecture

```python
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol, Any, overload

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')

# Column schema for structured data
ColumnSchema: TypeAlias = dict[str, type]

class Enumerable(ABC, Generic[T]):
    """Base class for all enumerable operations"""
    
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over elements"""
        pass
    
    # Core transformations
    def select(self, selector: Callable[[T], U]) -> 'Enumerable[U]':
        """Project each element (map)"""
        return Select(self, selector)
    
    def where(self, predicate: Callable[[T], bool]) -> 'Enumerable[T]':
        """Filter elements"""
        return Where(self, predicate)
    
    def select_many(self, selector: Callable[[T], Iterable[U]]) -> 'Enumerable[U]':
        """Flatten nested sequences"""
        return SelectMany(self, selector)
    
    # Aggregations (terminal operations)
    def count(self, predicate: Callable[[T], bool] | None = None) -> int:
        """Count elements matching predicate"""
        if predicate:
            return sum(1 for item in self if predicate(item))
        return sum(1 for _ in self)
    
    def first(self, predicate: Callable[[T], bool] | None = None) -> T:
        """Get first element"""
        for item in self:
            if predicate is None or predicate(item):
                return item
        raise ValueError("Sequence contains no matching element")
    
    def first_or_default(self, predicate: Callable[[T], bool] | None = None, 
                        default: T | None = None) -> T | None:
        """Get first element or default"""
        for item in self:
            if predicate is None or predicate(item):
                return item
        return default
    
    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Check if any element matches"""
        for item in self:
            if predicate is None or predicate(item):
                return True
        return False
    
    def all(self, predicate: Callable[[T], bool]) -> bool:
        """Check if all elements match"""
        for item in self:
            if not predicate(item):
                return False
        return True
    
    # Set operations
    def distinct(self) -> 'Enumerable[T]':
        """Remove duplicates"""
        return Distinct(self)
    
    def union(self, other: 'Enumerable[T]') -> 'Enumerable[T]':
        """Union with another enumerable"""
        return Union(self, other)
    
    def intersect(self, other: 'Enumerable[T]') -> 'Enumerable[T]':
        """Intersection with another enumerable"""
        return Intersect(self, other)
    
    def except_(self, other: 'Enumerable[T]') -> 'Enumerable[T]':
        """Difference from another enumerable"""
        return Except(self, other)
    
    # Ordering
    def order_by(self, key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """Order by key ascending"""
        return OrderBy(self, key_selector)
    
    def order_by_descending(self, key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """Order by key descending"""
        return OrderByDescending(self, key_selector)
    
    # Grouping
    def group_by(self, key_selector: Callable[[T], K]) -> 'Enumerable[Grouping[K, T]]':
        """Group by key"""
        return GroupBy(self, key_selector)
    
    # Joining
    def join(self, inner: 'Enumerable[U]', 
             outer_key_selector: Callable[[T], K],
             inner_key_selector: Callable[[U], K],
             result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """Inner join"""
        return Join(self, inner, outer_key_selector, inner_key_selector, result_selector)
    
    # Partitioning
    def take(self, count: int) -> 'Enumerable[T]':
        """Take first n elements"""
        return Take(self, count)
    
    def skip(self, count: int) -> 'Enumerable[T]':
        """Skip first n elements"""
        return Skip(self, count)
    
    def take_while(self, predicate: Callable[[T], bool]) -> 'Enumerable[T]':
        """Take while predicate is true"""
        return TakeWhile(self, predicate)
    
    def skip_while(self, predicate: Callable[[T], bool]) -> 'Enumerable[T]':
        """Skip while predicate is true"""
        return SkipWhile(self, predicate)
    
    # Windowing
    def window(self, size: int, step: int = 1) -> 'Enumerable[list[T]]':
        """Create sliding windows"""
        return Window(self, size, step)
    
    def batch(self, size: int) -> 'Enumerable[list[T]]':
        """Batch into chunks"""
        return Batch(self, size)
    
    # Materialization
    def to_list(self) -> list[T]:
        """Materialize to list"""
        return list(self)
    
    def to_dict(self, key_selector: Callable[[T], K], 
                value_selector: Callable[[T], V] | None = None) -> dict[K, V]:
        """Materialize to dictionary"""
        if value_selector is None:
            return {key_selector(item): item for item in self}  # type: ignore
        return {key_selector(item): value_selector(item) for item in self}
    
    def to_set(self) -> set[T]:
        """Materialize to set"""
        return set(self)

# Concrete implementations
class Select(Enumerable[U], Generic[T, U]):
    """Deferred select/map operation"""
    def __init__(self, source: Enumerable[T], selector: Callable[[T], U]):
        self.source = source
        self.selector = selector
    
    def __iter__(self) -> Iterator[U]:
        for item in self.source:
            yield self.selector(item)

class Where(Enumerable[T]):
    """Deferred filter operation"""
    def __init__(self, source: Enumerable[T], predicate: Callable[[T], bool]):
        self.source = source
        self.predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        for item in self.source:
            if self.predicate(item):
                yield item

class Window(Enumerable[list[T]], Generic[T]):
    """Sliding window operation"""
    def __init__(self, source: Enumerable[T], size: int, step: int = 1):
        self.source = source
        self.size = size
        self.step = step
    
    def __iter__(self) -> Iterator[list[T]]:
        window: list[T] = []
        iterator = iter(self.source)
        
        # Fill initial window
        for _ in range(self.size):
            try:
                window.append(next(iterator))
            except StopIteration:
                if window:
                    yield window
                return
        
        yield list(window)
        
        # Slide window
        while True:
            try:
                for _ in range(self.step):
                    window.pop(0)
                    window.append(next(iterator))
                yield list(window)
            except StopIteration:
                break

# Table implementation using Enumerable
@dataclass
class Table(Enumerable[T], Generic[T]):
    """Table implementation based on Enumerable"""
    _rows: list[T]
    _columns: ColumnSchema
    
    def __iter__(self) -> Iterator[T]:
        return iter(self._rows)
    
    def __len__(self) -> int:
        return len(self._rows)
    
    def __getitem__(self, index: int) -> T:
        return self._rows[index]
    
    @property
    def columns(self) -> ColumnSchema:
        return self._columns
    
    @staticmethod
    def from_rows(rows: list[T], columns: ColumnSchema | None = None) -> 'Table[T]':
        """Create table from rows"""
        if columns is None and rows and hasattr(rows[0], 'model_fields'):
            # Infer from Pydantic model
            fields = {k: v.annotation for k, v in type(rows[0]).model_fields.items()}
            columns = fields
        return Table(_rows=rows, _columns=columns or {})
    
    @staticmethod
    def empty() -> 'Table[Any]':
        """Create empty table"""
        return Table(_rows=[], _columns={})
    
    # Override to return Table types for fluent interface
    def where(self, predicate: Callable[[T], bool]) -> 'Table[T]':
        filtered = list(Where(self, predicate))
        return Table(_rows=filtered, _columns=self._columns)
    
    def select(self, selector: Callable[[T], U]) -> 'Table[U]':
        mapped = list(Select(self, selector))
        # Try to infer new columns
        new_columns = self._infer_columns(mapped) if mapped else {}
        return Table(_rows=mapped, _columns=new_columns)
    
    def _infer_columns(self, rows: list[Any]) -> ColumnSchema:
        """Infer column schema from rows"""
        if not rows:
            return {}
        
        first = rows[0]
        if hasattr(first, 'model_fields'):
            # Pydantic model
            return {k: v.annotation for k, v in type(first).model_fields.items()}
        elif isinstance(first, dict):
            # Dictionary rows
            return {k: type(v) for k, v in first.items()}
        else:
            # Can't infer
            return {}

# Convenience factory functions
def from_iterable(items: Iterable[T]) -> Enumerable[T]:
    """Create enumerable from any iterable"""
    return IterableEnumerable(items)

class IterableEnumerable(Enumerable[T]):
    """Wrapper for standard iterables"""
    def __init__(self, items: Iterable[T]):
        self.items = items
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

# Type registration for serialization
from tidyllm.serialization import SerializationRegistry

def _serialize_table(table: Table) -> dict:
    return {
        "columns": table.columns,
        "rows": [SerializationRegistry.serialize(row) for row in table]
    }

def _deserialize_table(data: dict) -> Table:
    return Table.from_rows(data["rows"], data["columns"])

SerializationRegistry.register_serializer(Table, _serialize_table)
SerializationRegistry.register_deserializer(Table, _deserialize_table)
```

### Additional Operations to Implement

1. **Aggregations**:
   - `sum()`, `average()`, `min()`, `max()`
   - `aggregate()` with custom accumulator

2. **Set Operations**:
   - Full implementations for `Distinct`, `Union`, `Intersect`, `Except`

3. **Ordering**:
   - `OrderedEnumerable` with `then_by()` support
   - Stable sort guarantees

4. **Grouping**:
   - `Grouping[K, T]` protocol
   - `group_join()` for left outer joins

5. **Advanced Windowing**:
   - `window_by()` with custom boundaries
   - `tumbling_window()` for non-overlapping windows

6. **Parallel Operations** (future):
   - `as_parallel()` for parallel execution
   - Thread-safe aggregations

## Migration Plan

### Phase 1: Create New Modules
1. Create `serialization.py` with all serialization logic
2. Create `function_schema.py` with function handling
3. Create `linq.py` with enumerable implementation

### Phase 2: Update Imports
1. Update all imports from `tidyllm.data` to appropriate new modules
2. Update type hints to use new `Enumerable` base class
3. Replace `ConcreteTable` with `Table` from `linq.py`

### Phase 3: Enhance Functionality
1. Add comprehensive LINQ operations
2. Implement deferred evaluation patterns
3. Add type-safe windowing operations

### Phase 4: Testing
1. Create comprehensive test suite for LINQ operations
2. Test deferred evaluation behavior
3. Performance benchmarks for large datasets

## Backwards Compatibility

Update all users, no backwards compat.

## Example Usage

```python
from tidyllm.linq import Table, from_iterable
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

# Create table
people = Table.from_rows([
    Person(name="Alice", age=30, city="NYC"),
    Person(name="Bob", age=25, city="LA"),
    Person(name="Charlie", age=35, city="NYC"),
])

# LINQ-style queries with deferred evaluation
query = (people
    .where(lambda p: p.age > 25)
    .select(lambda p: {"name": p.name, "city": p.city})
    .order_by(lambda p: p["name"]))

# Nothing executed yet - query is just an operation graph

# Materialize results
results = query.to_list()  # Now execution happens

# Windowing example
numbers = from_iterable(range(10))
windows = numbers.window(3, step=1).to_list()
# [[0, 1, 2], [1, 2, 3], [2, 3, 4], ...]

# Grouping example
by_city = (people
    .group_by(lambda p: p.city)
    .select(lambda g: {
        "city": g.key,
        "count": g.count(),
        "avg_age": g.select(lambda p: p.age).average()
    }))
```

## Benefits

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Type Safety**: Strong typing throughout with generic support
3. **Performance**: Deferred evaluation minimizes memory usage
4. **Flexibility**: Easy to extend with new operations
5. **Familiarity**: LINQ-style API is well-known and documented
6. **Testability**: Each operation can be tested in isolation
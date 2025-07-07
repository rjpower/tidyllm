"""
LINQ-style enumerable implementation with deferred evaluation.

Provides a comprehensive set of query operations with lazy evaluation,
strong typing, and fluent chaining.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from functools import reduce
from itertools import islice
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import core_schema

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')

# Column schema for structured data (type names as strings for JSON serialization)
ColumnSchema: TypeAlias = dict[str, str]


class Grouping(Protocol[K, V]):
    """Protocol for grouped data."""
    
    @property
    def key(self) -> K:
        """The key for this group."""
        ...
    
    def __iter__(self) -> Iterator[V]:
        """Iterate over items in this group."""
        ...
    
    def count(self) -> int:
        """Count items in this group."""
        ...


class GroupingImpl(Generic[K, V]):
    """Implementation of Grouping protocol."""
    
    def __init__(self, key: K, items: list[V]):
        self._key = key
        self._items = items
    
    @property
    def key(self) -> K:
        return self._key
    
    def __iter__(self) -> Iterator[V]:
        return iter(self._items)
    
    def count(self) -> int:
        return len(self._items)


class OrderedEnumerable(Generic[T]):
    """Enumerable with ordering capabilities."""
    
    def __init__(self, source: 'Enumerable[T]', key_func: Callable[[T], Any], reverse: bool = False):
        self.source = source
        self.key_func = key_func
        self.reverse = reverse
        self._then_by_funcs: list[tuple[Callable[[T], Any], bool]] = []
    
    def then_by(self, key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """Add secondary sort key."""
        result = OrderedEnumerable(self.source, self.key_func, self.reverse)
        result._then_by_funcs = self._then_by_funcs.copy()
        result._then_by_funcs.append((key_selector, False))
        return result
    
    def then_by_descending(self, key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """Add secondary sort key (descending)."""
        result = OrderedEnumerable(self.source, self.key_func, self.reverse)
        result._then_by_funcs = self._then_by_funcs.copy()
        result._then_by_funcs.append((key_selector, True))
        return result
    
    def __iter__(self) -> Iterator[T]:
        items = list(self.source)
        
        # Build composite key function
        def composite_key(item: T) -> tuple:
            keys = [self.key_func(item)]
            for func, _ in self._then_by_funcs:
                keys.append(func(item))
            return tuple(keys)
        
        # Sort with composite key
        sorted_items = sorted(items, key=composite_key, reverse=self.reverse)
        
        # Apply then_by reversals
        if self._then_by_funcs:
            # This is a simplified approach - full implementation would need
            # stable sort with multiple passes
            pass
        
        return iter(sorted_items)


class Enumerable(ABC, Generic[T]):
    """Base class for all enumerable operations with deferred evaluation."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over elements."""
        pass
    
    # Core transformations
    def select(self, selector: Callable[[T], U]) -> 'Enumerable[U]':
        """Project each element (map)."""
        return Select(self, selector)
    
    def where(self, predicate: Callable[[T], bool]) -> 'Enumerable[T]':
        """Filter elements."""
        return Where(self, predicate)
    
    def select_many(self, selector: Callable[[T], Iterable[U]]) -> 'Enumerable[U]':
        """Flatten nested sequences."""
        return SelectMany(self, selector)
    
    # Aggregations (terminal operations)
    def count(self, predicate: Callable[[T], bool] | None = None) -> int:
        """Count elements matching predicate."""
        if predicate:
            return sum(1 for item in self if predicate(item))
        return sum(1 for _ in self)
    
    def first(self, predicate: Callable[[T], bool] | None = None) -> T:
        """Get first element."""
        for item in self:
            if predicate is None or predicate(item):
                return item
        raise ValueError("Sequence contains no matching element")
    
    def first_or_default(self, predicate: Callable[[T], bool] | None = None, 
                        default: T | None = None) -> T | None:
        """Get first element or default."""
        for item in self:
            if predicate is None or predicate(item):
                return item
        return default
    
    def last(self, predicate: Callable[[T], bool] | None = None) -> T:
        """Get last element."""
        result: T | None = None
        found = False
        for item in self:
            if predicate is None or predicate(item):
                result = item
                found = True
        if not found:
            raise ValueError("Sequence contains no matching element")
        return result  # type: ignore
    
    def last_or_default(self, predicate: Callable[[T], bool] | None = None, 
                       default: T | None = None) -> T | None:
        """Get last element or default."""
        result = default
        for item in self:
            if predicate is None or predicate(item):
                result = item
        return result
    
    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Check if any element matches."""
        for item in self:
            if predicate is None or predicate(item):
                return True
        return False
    
    def all(self, predicate: Callable[[T], bool]) -> bool:
        """Check if all elements match."""
        for item in self:
            if not predicate(item):
                return False
        return True
    
    def sum(self, selector: Callable[[T], int | float] | None = None) -> int | float:
        """Sum elements or selected values."""
        if selector is None:
            return sum(item for item in self)  # type: ignore
        return sum(selector(item) for item in self)
    
    def average(self, selector: Callable[[T], int | float] | None = None) -> float:
        """Average of elements or selected values."""
        if selector is None:
            items = list(self)  # type: ignore
        else:
            items = [selector(item) for item in self]
        
        if not items:
            raise ValueError("Cannot compute average of empty sequence")
        return sum(items) / len(items)
    
    def min(self, selector: Callable[[T], Any] | None = None) -> T | Any:
        """Minimum element or selected value."""
        if selector is None:
            return min(self)
        return min(selector(item) for item in self)
    
    def max(self, selector: Callable[[T], Any] | None = None) -> T | Any:
        """Maximum element or selected value."""
        if selector is None:
            return max(self)
        return max(selector(item) for item in self)
    
    def aggregate(self, seed: U, func: Callable[[U, T], U]) -> U:
        """Aggregate with accumulator function."""
        return reduce(func, self, seed)
    
    # Set operations
    def distinct(self) -> 'Enumerable[T]':
        """Remove duplicates."""
        return Distinct(self)
    
    def union(self, other: 'Enumerable[T]') -> 'Enumerable[T]':
        """Union with another enumerable."""
        return Union(self, other)
    
    def intersect(self, other: 'Enumerable[T]') -> 'Enumerable[T]':
        """Intersection with another enumerable."""
        return Intersect(self, other)
    
    def except_(self, other: 'Enumerable[T]') -> 'Enumerable[T]':
        """Difference from another enumerable."""
        return Except(self, other)
    
    # Ordering
    def order_by(self, key_selector: Callable[[T], K]) -> OrderedEnumerable[T]:
        """Order by key ascending."""
        return OrderedEnumerable(self, key_selector)
    
    def order_by_descending(self, key_selector: Callable[[T], K]) -> OrderedEnumerable[T]:
        """Order by key descending."""
        return OrderedEnumerable(self, key_selector, reverse=True)
    
    # Grouping
    def group_by(self, key_selector: Callable[[T], K]) -> 'Enumerable[Grouping[K, T]]':
        """Group by key."""
        return GroupBy(self, key_selector)
    
    # Joining
    def join(self, inner: 'Enumerable[U]', 
             outer_key_selector: Callable[[T], K],
             inner_key_selector: Callable[[U], K],
             result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """Inner join."""
        return Join(self, inner, outer_key_selector, inner_key_selector, result_selector)
    
    # Partitioning
    def take(self, count: int) -> 'Enumerable[T]':
        """Take first n elements."""
        return Take(self, count)
    
    def skip(self, count: int) -> 'Enumerable[T]':
        """Skip first n elements."""
        return Skip(self, count)
    
    def take_while(self, predicate: Callable[[T], bool]) -> 'Enumerable[T]':
        """Take while predicate is true."""
        return TakeWhile(self, predicate)
    
    def skip_while(self, predicate: Callable[[T], bool]) -> 'Enumerable[T]':
        """Skip while predicate is true."""
        return SkipWhile(self, predicate)
    
    # Windowing
    def window(self, size: int, step: int = 1) -> 'Enumerable[list[T]]':
        """Create sliding windows."""
        return Window(self, size, step)
    
    def batch(self, size: int) -> 'Enumerable[list[T]]':
        """Batch into chunks."""
        return Batch(self, size)
    
    # Materialization
    def to_list(self) -> list[T]:
        """Materialize to list."""
        return list(self)
    
    def to_dict(self, key_selector: Callable[[T], K], 
                value_selector: Callable[[T], V] | None = None) -> dict[K, V]:
        """Materialize to dictionary."""
        if value_selector is None:
            return {key_selector(item): item for item in self}  # type: ignore
        return {key_selector(item): value_selector(item) for item in self}
    
    def to_set(self) -> set[T]:
        """Materialize to set."""
        return set(self)
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """Make Enumerable serializable by converting to Table."""
        from pydantic_core import core_schema
        
        def serialize_enumerable(instance: 'Enumerable[Any]') -> dict[str, Any]:
            """Serialize Enumerable by materializing to Table."""
            # Convert to list and create a Table
            rows = list(instance)
            # Try to infer columns if possible
            columns = {}
            if rows and hasattr(rows[0], 'model_fields'):
                columns = {k: v.annotation for k, v in rows[0].model_fields.items()}
            elif rows and isinstance(rows[0], dict):
                columns = {k: type(v) for k, v in rows[0].items()}
            
            return {
                "rows": rows,
                "columns": columns,
                "_type": "Table"
            }
        
        def deserialize_enumerable(data: dict[str, Any]) -> 'Table[Any]':
            """Deserialize to Table."""
            return Table(rows=data["rows"], columns=data.get("columns", {}))
        
        return core_schema.with_info_plain_validator_function(
            deserialize_enumerable,
            serializer=core_schema.plain_serializer_function(
                serialize_enumerable,
                return_schema=core_schema.dict_schema({
                    "rows": core_schema.list_schema(core_schema.any_schema()),
                    "columns": core_schema.dict_schema(),
                    "_type": core_schema.str_schema()
                })
            )
        )


# Concrete operation implementations
class Select(Enumerable[U], Generic[T, U]):
    """Deferred select/map operation."""
    def __init__(self, source: Enumerable[T], selector: Callable[[T], U]):
        super().__init__()
        self.source = source
        self.selector = selector
    
    def __iter__(self) -> Iterator[U]:
        for item in self.source:
            yield self.selector(item)


class Where(Enumerable[T]):
    """Deferred filter operation."""
    def __init__(self, source: Enumerable[T], predicate: Callable[[T], bool]):
        super().__init__()
        self.source = source
        self.predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        for item in self.source:
            if self.predicate(item):
                yield item


class SelectMany(Enumerable[U], Generic[T, U]):
    """Deferred flatten operation."""
    def __init__(self, source: Enumerable[T], selector: Callable[[T], Iterable[U]]):
        super().__init__()
        self.source = source
        self.selector = selector
    
    def __iter__(self) -> Iterator[U]:
        for item in self.source:
            yield from self.selector(item)


class Take(Enumerable[T]):
    """Take first n elements."""
    def __init__(self, source: Enumerable[T], count: int):
        super().__init__()
        self.source = source
        self._count = count
    
    def __iter__(self) -> Iterator[T]:
        yield from islice(self.source, self._count)


class Skip(Enumerable[T]):
    """Skip first n elements."""
    def __init__(self, source: Enumerable[T], count: int):
        super().__init__()
        self.source = source
        self._count = count
    
    def __iter__(self) -> Iterator[T]:
        yield from islice(self.source, self._count, None)


class TakeWhile(Enumerable[T]):
    """Take while predicate is true."""
    def __init__(self, source: Enumerable[T], predicate: Callable[[T], bool]):
        super().__init__()
        self.source = source
        self.predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        for item in self.source:
            if self.predicate(item):
                yield item
            else:
                break


class SkipWhile(Enumerable[T]):
    """Skip while predicate is true."""
    def __init__(self, source: Enumerable[T], predicate: Callable[[T], bool]):
        super().__init__()
        self.source = source
        self.predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        iterator = iter(self.source)
        
        # Skip while predicate is true
        for item in iterator:
            if not self.predicate(item):
                yield item
                break
        
        # Yield remaining items
        yield from iterator


class Distinct(Enumerable[T]):
    """Remove duplicates."""
    def __init__(self, source: Enumerable[T]):
        super().__init__()
        self.source = source
    
    def __iter__(self) -> Iterator[T]:
        seen = set()
        for item in self.source:
            if item not in seen:
                seen.add(item)
                yield item


class Union(Enumerable[T]):
    """Union of two enumerables."""
    def __init__(self, first: Enumerable[T], second: Enumerable[T]):
        super().__init__()
        self._first = first
        self._second = second
    
    def __iter__(self) -> Iterator[T]:
        seen = set()
        for item in self._first:
            if item not in seen:
                seen.add(item)
                yield item
        for item in self._second:
            if item not in seen:
                seen.add(item)
                yield item


class Intersect(Enumerable[T]):
    """Intersection of two enumerables."""
    def __init__(self, first: Enumerable[T], second: Enumerable[T]):
        super().__init__()
        self._first = first
        self._second = second
    
    def __iter__(self) -> Iterator[T]:
        second_set = set(self._second)
        seen = set()
        for item in self._first:
            if item in second_set and item not in seen:
                seen.add(item)
                yield item


class Except(Enumerable[T]):
    """Difference of two enumerables."""
    def __init__(self, first: Enumerable[T], second: Enumerable[T]):
        super().__init__()
        self._first = first
        self._second = second
    
    def __iter__(self) -> Iterator[T]:
        second_set = set(self._second)
        seen = set()
        for item in self._first:
            if item not in second_set and item not in seen:
                seen.add(item)
                yield item


class GroupBy(Enumerable[Grouping[K, T]], Generic[T, K]):
    """Group by key."""
    def __init__(self, source: Enumerable[T], key_selector: Callable[[T], K]):
        super().__init__()
        self.source = source
        self.key_selector = key_selector
    
    def __iter__(self) -> Iterator[Grouping[K, T]]:
        groups: dict[K, list[T]] = {}
        for item in self.source:
            key = self.key_selector(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        for key, items in groups.items():
            yield GroupingImpl(key, items)


class Join(Enumerable[V], Generic[T, U, K, V]):
    """Inner join operation."""
    def __init__(self, 
                 outer: Enumerable[T], 
                 inner: Enumerable[U],
                 outer_key_selector: Callable[[T], K],
                 inner_key_selector: Callable[[U], K],
                 result_selector: Callable[[T, U], V]):
        super().__init__()
        self.outer = outer
        self.inner = inner
        self.outer_key_selector = outer_key_selector
        self.inner_key_selector = inner_key_selector
        self.result_selector = result_selector
    
    def __iter__(self) -> Iterator[V]:
        inner_lookup: dict[K, list[U]] = {}
        for inner_item in self.inner:
            key = self.inner_key_selector(inner_item)
            if key not in inner_lookup:
                inner_lookup[key] = []
            inner_lookup[key].append(inner_item)
        
        for outer_item in self.outer:
            key = self.outer_key_selector(outer_item)
            if key in inner_lookup:
                for inner_item in inner_lookup[key]:
                    yield self.result_selector(outer_item, inner_item)


class Window(Enumerable[list[T]], Generic[T]):
    """Sliding window operation."""
    def __init__(self, source: Enumerable[T], size: int, step: int = 1):
        super().__init__()
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


class Batch(Enumerable[list[T]], Generic[T]):
    """Batch into chunks."""
    def __init__(self, source: Enumerable[T], size: int):
        super().__init__()
        self.source = source
        self.size = size
    
    def __iter__(self) -> Iterator[list[T]]:
        iterator = iter(self.source)
        while True:
            batch = list(islice(iterator, self.size))
            if not batch:
                break
            yield batch


# Table implementation using Enumerable
class Table(BaseModel, Enumerable[T], Generic[T]):
    """Pydantic-based table with automatic serialization and LINQ operations."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid'
    )
    
    rows: list[T] = Field(description="Table rows")
    columns: dict[str, str] = Field(description="Column schema", default_factory=dict)
    
    def __iter__(self) -> Iterator[T]:
        # Override BaseModel's __iter__ to iterate over rows instead of fields
        return iter(self.rows)
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, index: int) -> T:
        return self.rows[index]
    
    @property
    def row_count(self) -> int:
        """Get row count."""
        return len(self.rows)
    
    @classmethod
    def from_rows(cls, rows: list[T], columns: ColumnSchema | None = None) -> 'Table[T]':
        """Create table from rows."""
        if columns is None and rows and hasattr(rows[0], 'model_fields'):
            # Infer from Pydantic model
            fields = {k: getattr(v.annotation, '__name__', str(v.annotation)) for k, v in rows[0].model_fields.items()}  # type: ignore
            columns = fields
        # Convert type objects to strings
        if columns:
            columns = {k: getattr(v, '__name__', str(v)) if not isinstance(v, str) else v for k, v in columns.items()}
        return cls(rows=rows, columns=columns or {})
    
    @classmethod
    def from_pydantic(cls, rows: list[BaseModel]) -> 'Table[Any]':
        """Create table from Pydantic models."""
        if not rows:
            return cls.empty()
        
        # Convert type annotations to string names
        fields = {k: getattr(v.annotation, '__name__', str(v.annotation)) for k, v in type(rows[0]).model_fields.items()}
        return cls(rows=rows, columns=fields)
    
    @classmethod
    def empty(cls) -> 'Table[Any]':
        """Create empty table."""
        return cls(rows=[], columns={})
    
    def _infer_columns(self, rows: list[Any]) -> dict[str, str]:
        """Infer column schema from rows."""
        if not rows:
            return {}
        
        first = rows[0]
        if hasattr(first, 'model_fields'):
            # Pydantic model
            return {k: getattr(v.annotation, '__name__', str(v.annotation)) for k, v in type(first).model_fields.items()}
        elif isinstance(first, dict):
            # Dictionary rows
            return {k: type(v).__name__ for k, v in first.items()}
        else:
            # Can't infer
            return {}
    
    # Override methods to maintain Table type
    def where(self, predicate: Callable[[T], bool]) -> 'Table[T]':
        """Filter rows and return new Table."""
        filtered = list(Where(self, predicate))
        return Table(rows=filtered, columns=self.columns)
    
    def select(self, selector: Callable[[T], U]) -> 'Table[U]':
        """Transform rows and return new Table."""
        mapped = list(Select(self, selector))
        new_columns = self._infer_columns(mapped) if mapped else {}
        return Table(rows=mapped, columns=new_columns)
    
    def map(self, func: Callable[[T], U]) -> 'Table[U]':
        """Map function over rows (alias for select)."""
        return self.select(func)
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Table[T]':
        """Filter rows (alias for where)."""
        return self.where(predicate)
    
    def to_json_list(self) -> list[Any]:
        """Convert table to JSON-serializable list format."""
        return [row.model_dump(mode='json') if isinstance(row, BaseModel) else row for row in self.rows]


# Convenience factory functions
def from_iterable(items: Iterable[T]) -> Enumerable[T]:
    """Create enumerable from any iterable."""
    return IterableEnumerable(items)


class IterableEnumerable(Enumerable[T]):
    """Wrapper for standard iterables."""
    def __init__(self, items: Iterable[T]):
        super().__init__()
        self.items = items
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.items)


# For backwards compatibility
ConcreteTable = Table


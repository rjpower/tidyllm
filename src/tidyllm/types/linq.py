"""
LINQ-style enumerable implementation with deferred evaluation.

Provides a comprehensive set of query operations with lazy evaluation,
strong typing, and fluent chaining.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from functools import reduce
from itertools import islice
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_core import core_schema

from tidyllm.types.serialization import create_model_from_data_sample

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
V = TypeVar("V")


class Grouping(Generic[K, V]):
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


class Enumerable(ABC, Generic[T]):
    """Base class for all enumerable operations with deferred evaluation."""

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over elements."""
        pass

    # Core transformations
    def select(self, selector: Callable[[T], U]) -> "Enumerable[U]":
        """Project each element (map)."""
        return Select(self, selector)

    def where(self, predicate: Callable[[T], bool]) -> "Enumerable[T]":
        """Filter elements."""
        return Where(self, predicate)

    def select_many(self, selector: Callable[[T], Iterable[U]]) -> "Enumerable[U]":
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

    def first_or_default(
        self, predicate: Callable[[T], bool] | None = None, default: T | None = None
    ) -> T | None:
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

    def last_or_default(
        self, predicate: Callable[[T], bool] | None = None, default: T | None = None
    ) -> T | None:
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

    def sum(self) -> int | float:
        """Sum numeric elements. Use .select() first to transform if needed."""
        return sum(item for item in self)  # type: ignore

    def average(self) -> float:
        """Average of numeric elements. Use .select() first to transform if needed."""
        total = 0.0
        count = 0
        for item in self:
            total += item  # type: ignore
            count += 1

        if count == 0:
            raise ValueError("Cannot compute average of empty sequence")
        return total / count

    def min(self) -> T:
        """Minimum element. Use .select() first to transform if needed."""
        return min(self)

    def max(self) -> T:
        """Maximum element. Use .select() first to transform if needed."""
        return max(self)

    def aggregate(self, seed: U, func: Callable[[U, T], U]) -> U:
        """Aggregate with accumulator function."""
        return reduce(func, self, seed)

    # Set operations
    def distinct(self) -> "Enumerable[T]":
        """Remove duplicates."""
        return Distinct(self)

    def union(self, other: "Enumerable[T]") -> "Enumerable[T]":
        """Union with another enumerable."""
        return Union(self, other)

    def intersect(self, other: "Enumerable[T]") -> "Enumerable[T]":
        """Intersection with another enumerable."""
        return Intersect(self, other)

    def except_(self, other: "Enumerable[T]") -> "Enumerable[T]":
        """Difference from another enumerable."""
        return Except(self, other)

    # Ordering
    def order_by(self, key_selector: Callable[[T], K]) -> "OrderedEnumerable[T]":
        """Order by key ascending."""
        return OrderedEnumerable(self, key_selector)

    def order_by_descending(
        self, key_selector: Callable[[T], K]
    ) -> "OrderedEnumerable[T]":
        """Order by key descending."""
        return OrderedEnumerable(self, key_selector, reverse=True)

    # Grouping
    def group_by(self, key_selector: Callable[[T], K]) -> "Enumerable[Grouping[K, T]]":
        """Group by key."""
        return GroupBy(self, key_selector)

    # Joining
    def join(
        self,
        inner: "Enumerable[U]",
        outer_key_selector: Callable[[T], K],
        inner_key_selector: Callable[[U], K],
        result_selector: Callable[[T, U], V],
    ) -> "Enumerable[V]":
        """Inner join."""
        return Join(
            self, inner, outer_key_selector, inner_key_selector, result_selector
        )

    # Partitioning
    def take(self, count: int) -> "Enumerable[T]":
        """Take first n elements."""
        return Take(self, count)

    def skip(self, count: int) -> "Enumerable[T]":
        """Skip first n elements."""
        return Skip(self, count)

    def take_while(self, predicate: Callable[[T], bool]) -> "Enumerable[T]":
        """Take while predicate is true."""
        return TakeWhile(self, predicate)

    def skip_while(self, predicate: Callable[[T], bool]) -> "Enumerable[T]":
        """Skip while predicate is true."""
        return SkipWhile(self, predicate)

    # Windowing
    def window(
        self,
        size_or_predicate: int | Callable[[list[T]], bool],
        step: int = 1,
    ) -> "Enumerable[list[T]]":
        """Create sliding windows by size or predicate."""
        if isinstance(size_or_predicate, int):
            return Window(self, size_or_predicate, step)
        else:
            return WindowPredicate(self, size_or_predicate)

    def batch(self, size: int) -> "Enumerable[list[T]]":
        """Batch into chunks."""
        return Batch(self, size)

    def split(self, n: int = 2) -> list["Enumerable[T]"]:
        """Split enumerable into n independent copies."""
        items = self.to_list()
        return [from_iterable(items) for _ in range(n)]

    def reduce(
        self,
        reducer: Callable[[U, T], U],
        initial: U,
    ) -> U:
        """Reduce enumerable to single value using accumulator function."""
        return self.aggregate(initial, reducer)

    def sink(self, sink_func: Callable[[T], Any]) -> None:
        """Consume enumerable with sink function."""
        for item in self:
            sink_func(item)

    def collect(self) -> list[T]:
        """Collect all elements into list (alias for to_list)."""
        return self.to_list()

    # Materialization
    def to_list(self) -> list[T]:
        """Materialize to list."""
        return list(self)

    def to_dict(
        self,
        key_selector: Callable[[T], K],
        value_selector: Callable[[T], V] | None = None,
    ) -> dict[K, V]:
        """Materialize to dictionary."""
        if value_selector is None:
            return {key_selector(item): item for item in self}  # type: ignore
        return {key_selector(item): value_selector(item) for item in self}

    def to_set(self) -> set[T]:
        """Materialize to set."""
        return set(self)

    def partition(
        self, predicate: Callable[[T], bool]
    ) -> tuple["Enumerable[T]", "Enumerable[T]"]:
        """Split into two enumerables based on predicate.

        Args:
            predicate: Function to test each element

        Returns:
            Tuple of (matching elements, non-matching elements)
        """
        matching = []
        non_matching = []
        for item in self:
            if predicate(item):
                matching.append(item)
            else:
                non_matching.append(item)
        return from_iterable(matching), from_iterable(non_matching)

    def try_select(
        self, selector: Callable[[T], U]
    ) -> tuple["Enumerable[U]", "Enumerable[Exception]"]:
        """Select with automatic exception handling, returning successes and exceptions separately.

        Args:
            selector: Function to transform each element

        Returns:
            Tuple of (successful results, exceptions)
        """
        successes = []
        errors = []
        for item in self:
            try:
                result = selector(item)
                successes.append(result)
            except Exception as e:
                errors.append(e)
        return from_iterable(successes), from_iterable(errors)

    def with_progress(self, description: str = "Processing") -> "Enumerable[T]":
        """Add Rich progress tracking to enumeration.

        Args:
            description: Description to show in progress bar

        Returns:
            Enumerable that displays progress during iteration
        """
        return WithProgress(self, description)

    def to_lookup(self, key_selector: Callable[[T], K]) -> dict[K, list[T]]:
        """Create dictionary grouped by key.

        Args:
            key_selector: Function to extract grouping key

        Returns:
            Dictionary mapping keys to lists of items
        """
        lookup: dict[K, list[T]] = {}
        for item in self:
            key = key_selector(item)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(item)
        return lookup

    def index_by(self, key_selector: Callable[[T], K]) -> dict[K, T]:
        """Create dictionary indexed by unique key.

        Args:
            key_selector: Function to extract unique key

        Returns:
            Dictionary mapping keys to single items

        Raises:
            ValueError: If duplicate keys are found
        """
        index: dict[K, T] = {}
        for item in self:
            key = key_selector(item)
            if key in index:
                raise ValueError(f"Duplicate key found: {key}")
            index[key] = item
        return index

    def to_table(self) -> "Table":
        """Compute the full values of this enumerable and return as a fixed table."""
        values = list(self)
        return Table.from_rows(values)

    def with_schema_inference(
        self, sample_size: int = 5
    ) -> "SchemaInferringEnumerable[T]":
        """Enable schema inference for this enumerable.

        Args:
            sample_size: Number of items to sample for schema inference

        Returns:
            Schema-inferring wrapper that can provide type information
        """
        return SchemaInferringEnumerable(self, sample_size)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """Serialization logic for Enumerable.

        Enumerables are serialized by first materializing them to `Table` form
        then inferring the underlying model type for the schema.
        """

        def serialize_enumerable(instance: "Enumerable[Any]") -> dict[str, Any]:
            """Serialize Enumerable by materializing to Table."""
            table = instance.to_table()
            schema = table.table_schema().model_json_schema()

            return {"rows": table.rows, "table_schema": schema, "_type": "Table"}

        def deserialize_enumerable(data: Any) -> Any:
            """Deserialize to Table or pass through if already an Enumerable."""
            if isinstance(data, Enumerable):
                return data
            if isinstance(data, dict) and "rows" in data:
                # For now, create Table without schema - it will infer from rows
                return Table(rows=data["rows"], table_schema=None)
            return data

        # Create an any schema that accepts Enumerable instances
        return core_schema.no_info_before_validator_function(
            deserialize_enumerable,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_enumerable,
                info_arg=False,
                return_schema=core_schema.dict_schema(),
            ),
        )


class OrderedEnumerable(Enumerable[T]):
    """Enumerable with ordering capabilities."""

    def __init__(
        self,
        source: "Enumerable[T]",
        key_func: Callable[[T], Any],
        reverse: bool = False,
    ):
        super().__init__()
        self.source = source
        self.key_func = key_func
        self.reverse = reverse
        self._then_by_funcs: list[tuple[Callable[[T], Any], bool]] = []

    def then_by(self, key_selector: Callable[[T], K]) -> "OrderedEnumerable[T]":
        """Add secondary sort key."""
        result = OrderedEnumerable(self.source, self.key_func, self.reverse)
        result._then_by_funcs = self._then_by_funcs.copy()
        result._then_by_funcs.append((key_selector, False))
        return result

    def then_by_descending(
        self, key_selector: Callable[[T], K]
    ) -> "OrderedEnumerable[T]":
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
            yield Grouping(key, items)


class Join(Enumerable[V], Generic[T, U, K, V]):
    """Inner join operation."""

    def __init__(
        self,
        outer: Enumerable[T],
        inner: Enumerable[U],
        outer_key_selector: Callable[[T], K],
        inner_key_selector: Callable[[U], K],
        result_selector: Callable[[T, U], V],
    ):
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


class WithProgress(Enumerable[T]):
    """Add progress tracking to enumeration."""

    def __init__(self, source: Enumerable[T], description: str = "Processing"):
        super().__init__()
        self.source = source
        self.description = description

    def __iter__(self) -> Iterator[T]:
        try:
            from tqdm import tqdm

            for item in tqdm(self.source, desc=self.description):
                yield item
        except ImportError:
            # Fall back to regular iteration if rich is not available
            for item in self.source:
                yield item


class WindowPredicate(Enumerable[list[T]], Generic[T]):
    """Window by predicate condition."""

    def __init__(self, source: Enumerable[T], predicate: Callable[[list[T]], bool]):
        super().__init__()
        self.source = source
        self.predicate = predicate

    def __iter__(self) -> Iterator[list[T]]:
        window: list[T] = []
        for item in self.source:
            window.append(item)
            if self.predicate(window):
                yield list(window)
                window = []
        if window:
            yield window


class SchemaInferringEnumerable(Enumerable[T]):
    """Enumerable that infers schema from data samples while preserving lazy evaluation."""

    def __init__(self, source: Enumerable[T], sample_size: int = 5):
        super().__init__()
        self.source = source
        self.sample_size = sample_size
        self._cached_schema: type[BaseModel] | None = None
        self._sample_buffer: list[T] = []
        self._cached_iterator: Iterator[T] | None = None

    def set_known_schema(self, schema: type[BaseModel]) -> None:
        """Set a known schema, bypassing inference completely."""
        self._cached_schema = schema

    def table_schema(self) -> type[BaseModel]:
        """Get inferred schema, sampling if needed."""
        if self._cached_schema is not None:
            return self._cached_schema

        self._ensure_sampled()
        self._cached_schema = create_model_from_data_sample(
            self._sample_buffer, "InferredSchema"
        )
        return self._cached_schema

    def _ensure_sampled(self):
        """Collect sample if not already done."""
        if self._cached_schema:
            return

        # Get iterator from source and cache it
        self._cached_iterator = iter(self.source)

        # Collect sample from the cached iterator
        for _ in range(self.sample_size):
            try:
                item = next(self._cached_iterator)
                self._sample_buffer.append(item)
            except StopIteration:
                break

        self._sample_consumed = True

    def __iter__(self) -> Iterator[T]:
        """Iterate elegantly: yield sample buffer first, then continue with cached iterator."""
        self._ensure_sampled()

        # First, yield items from our sample buffer
        yield from self._sample_buffer

        # Then continue with the cached iterator (which is positioned after the sample)
        if self._cached_iterator is not None:
            yield from self._cached_iterator


# Table implementation inheriting from SchemaInferringEnumerable
class Table(SchemaInferringEnumerable[T]):
    """Table with automatic schema inference and LINQ operations."""

    def __init__(
        self, rows: list[T] | None = None, table_schema: type[BaseModel] | None = None
    ):
        self.rows: list[T] = rows or []

        source = IterableEnumerable(self.rows)
        super().__init__(source, sample_size=5)

        if table_schema is not None:
            self.set_known_schema(table_schema)

    def __iter__(self) -> Iterator[T]:
        """Direct iteration over rows (no sampling needed)."""
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
    def from_rows(
        cls, rows: list[T], table_schema: type[BaseModel] | None = None
    ) -> "Table[T]":
        """Create table from rows."""
        return cls(rows=rows, table_schema=table_schema)

    @classmethod
    def from_pydantic(cls, rows: list[BaseModel]) -> "Table[Any]":
        """Create table from Pydantic models with schema optimization."""
        if not rows:
            return cls.empty()

        model_type = type(rows[0])
        table = cls(rows=rows, table_schema=model_type)

        table.set_known_schema(model_type)
        return table

    @classmethod
    def empty(cls) -> "Table[Any]":
        """Create empty table."""
        return cls(rows=[], table_schema=None)

    def to_table(self) -> "Table[T]":
        return self


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

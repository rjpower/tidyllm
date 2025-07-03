from collections.abc import Callable, Iterable, Iterator
from itertools import tee
from typing import Any, Generic, TypeVar

StreamType = TypeVar("StreamType")
ResultType = TypeVar("ResultType")
ReducerType = TypeVar("ReducerType")


class Stream(Generic[StreamType]):
    """Type-safe stream with built-in operators."""

    _iterator: Iterator[StreamType]
    _cleanup: Callable[[], None] | None

    def __init__(
        self,
        iterator: Iterator[StreamType],
        cleanup: Callable[[], None] | None = None,
    ):
        self._iterator = iterator
        self._cleanup = cleanup

    def map(self, mapper: Callable[[StreamType], ResultType]) -> "Stream[ResultType]":
        """Apply transformation to each element."""
        return Stream(
            (mapper(item) for item in self._iterator),
            self._cleanup,
        )

    def filter(self, predicate: Callable[[StreamType], bool]) -> "Stream[StreamType]":
        """Filter elements based on predicate."""

        return Stream(
            (item for item in self._iterator if predicate(item)),
            self._cleanup,
        )

    def batch(self, size: int) -> "Stream[list[StreamType]]":
        """Group elements into batches."""

        def batch_iterator():
            batch = []
            for item in self._iterator:
                batch.append(item)
                if len(batch) >= size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        return Stream(batch_iterator(), self._cleanup)

    def window(
        self,
        size_or_predicate: int | Callable[[list[StreamType]], bool],
    ) -> "Stream[list[StreamType]]":
        """Window elements by count or condition."""
        predicate: Callable[[list[StreamType]], bool]
        if isinstance(size_or_predicate, int):
            predicate = lambda x: len(x) == size_or_predicate
        else:
            predicate = size_or_predicate

        def window_iterator():
            window = []
            for item in self._iterator:
                window.append(item)
                if predicate(window):
                    yield window
                    window = []

            if window:
                yield window

        return Stream(window_iterator(), self._cleanup)

    def split(self, n: int = 2) -> list["Stream[StreamType]"]:
        """Split into multiple streams using tee."""
        iterators = tee(self._iterator, n)
        return [Stream(it, self._cleanup) for it in iterators]

    def reduce(
        self,
        reducer: Callable[[ReducerType, StreamType], ReducerType],
        initial: ReducerType,
    ) -> ReducerType:
        """Reduce stream to single value."""

        result = initial
        for item in self._iterator:
            result = reducer(result, item)
        return result

    def sink(
        self,
        sink_func: Callable[[StreamType], Any],
    ) -> None:
        """Consume stream with a sink function."""

        for item in self._iterator:
            sink_func(item)

    def collect(self) -> list[StreamType]:
        """Collect all elements into a list."""
        return list(self._iterator)

    def take(self, n: int) -> "Stream[StreamType]":
        """Take first n elements."""

        def take_iterator():
            for count, item in enumerate(self._iterator):
                if count >= n:
                    break
                yield item

        return Stream(take_iterator(), self._cleanup)

    def cleanup(self) -> None:
        """Execute cleanup function if available."""
        if self._cleanup:
            self._cleanup()

    def __iter__(self) -> Iterator[StreamType]:
        return self._iterator

    def __enter__(self) -> "Stream[StreamType]":
        return self

    def __exit__(self, *args: Any):
        self.cleanup()


def create_stream_from_iterator(
    iterator_factory: Callable[[], Iterator[StreamType]],
    cleanup: Callable[[], None] | None = None,
) -> Stream[StreamType]:
    """Create a stream from an iterator factory with automatic context injection.

    This helper simplifies creating streams from iterators by:
    1. Getting the current context automatically
    2. Creating the iterator
    3. Wrapping it in a Stream with proper cleanup

    Args:
        iterator_factory: A callable that creates the iterator
        cleanup: Optional cleanup function

    Returns:
        A Stream wrapping the iterator
    """
    iterator = iterator_factory()
    return Stream(iterator=iterator, cleanup=cleanup)


def iterable_to_stream(iterable: Iterable[Any]) -> Stream[Any]:
    """Convert an iterable to a stream."""
    return Stream(iterable)


def stream_to_iterable(stream: Stream[Any]) -> Iterable[Any]:
    """Convert a stream to an iterable."""
    return stream._iterator
"""Function caching decorators.

This module provides decorators for caching function results in a database,
supporting both synchronous and asynchronous functions.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, runtime_checkable

from tidyllm.context import get_tool_context
from tidyllm.function_schema import FunctionDescription
from tidyllm.types.serialization import from_json_dict, to_json_str

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheDbProtocol(Protocol):
    """Protocol defining the minimal database interface required for caching."""

    def __getitem__(self, key: str) -> str:
        """Get cached result by key."""
        ...

    def __setitem__(self, key: str, value: str) -> None:
        """Store result in cache."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


class CacheContext(Protocol):
    @property
    def cache_db(self) -> CacheDbProtocol: ...


class DummyAdapter(CacheDbProtocol):
    def __getitem__(self, key: str):
        raise KeyError()

    def __setitem__(self, key: str, value: str):
        return

    def __contains__(self, key: str):
        return False


class SqlAdapter(CacheDbProtocol):
    def __init__(self, db: sqlite3.Connection, table_name: str = "function_cache"):
        self._db = db
        self._table_name = table_name
        self._setup_schema()

    def _setup_schema(self):
        """Setup the cache table schema."""
        cursor = self._db.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                arg_hash TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self._db.commit()

    def __getitem__(self, key: str) -> str:
        """Get cached result by argument hash."""
        cursor = self._db.cursor()
        cursor.execute(
            f"SELECT result FROM {self._table_name} WHERE arg_hash = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Cache key not found: {key}")
        return row[0]

    def __setitem__(self, key: str, value: str) -> None:
        """Store result in cache."""
        cursor = self._db.cursor()
        cursor.execute(
            f"INSERT OR REPLACE INTO {self._table_name} (arg_hash, result) VALUES (?, ?)",
            (key, value),
        )
        self._db.commit()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        cursor = self._db.cursor()
        cursor.execute(
            f"SELECT 1 FROM {self._table_name} WHERE arg_hash = ? LIMIT 1", (key,)
        )
        return cursor.fetchone() is not None


R = TypeVar("R")
P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class CacheResult(Generic[T]):
    """Result of a cache lookup operation."""

    exists: bool
    value: T | None = None

    @classmethod
    def hit(cls, value: T) -> "CacheResult[T]":
        """Create a cache hit result."""
        return cls(exists=True, value=value)

    @classmethod
    def miss(cls) -> "CacheResult[T]":
        """Create a cache miss result."""
        return cls(exists=False, value=None)


class _FunctionCacheHandler(Generic[P, R]):
    """Internal handler for function caching logic."""

    func: Callable[P, R]
    cache_db: CacheDbProtocol
    table_name: str
    description: FunctionDescription

    def __init__(self, func: Callable[P, R], cache_db: CacheDbProtocol):
        self.func = func
        self.cache_db = cache_db
        self.description = FunctionDescription(func)
        self.table_name = f"{self.description.name}_cache"

    def compute_arg_hash(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Compute a hash of the function arguments."""
        # Create argument model with original args first for validation
        args_instance = self.description.arg_model_from_args(*args, **kwargs)
        args_json = args_instance.model_dump_json()
        args_hash = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        return args_hash

    def lookup_cache(self, arg_hash: str) -> CacheResult[R]:
        """Single method to check cache and return result if exists."""
        if arg_hash in self.cache_db:
            result_json = self.cache_db[arg_hash]
            result_data = json.loads(result_json)
            print("Restoring from cache", self.description.result_type)
            parsed_result = from_json_dict(result_data, self.description.result_type)
            return CacheResult.hit(parsed_result)

        return CacheResult.miss()

    def store_result(self, arg_hash: str, result_data: R):
        """Store result in cache."""
        result_json = to_json_str(result_data)
        self.cache_db[arg_hash] = result_json


def cached_function(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for caching synchronous function results.
    
    Uses the current tool context to get cache configuration.

    Example:
        >>> @cached_function
        ... def expensive_computation(x: int) -> int:
        ...     return x * x
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            ctx = get_tool_context()
        except RuntimeError:
            # No cache context available, execute function directly
            return func(*args, **kwargs)

        handler = _FunctionCacheHandler(func, ctx.cache_db)
        arg_hash = handler.compute_arg_hash(*args, **kwargs)

        cache_result = handler.lookup_cache(arg_hash)
        if cache_result.exists:
            logger.debug(f"Cache hit for {func.__name__}")
            return cache_result.value  # type: ignore

        logger.debug(f"Cache miss for {func.__name__}")
        actual_result = func(*args, **kwargs)
        handler.store_result(arg_hash, actual_result)
        return actual_result

    return wrapper


def async_cached_function(
    func: Callable[P, Coroutine[Any, Any, R]]
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Decorator for caching asynchronous function results.
    
    Uses the current tool context to get cache configuration.

    Example:
        >>> @async_cached_function
        ... async def fetch_data(url: str) -> dict:
        ...     # expensive async operation
        ...     return {"data": "result"}
    """
    if not asyncio.iscoroutinefunction(func):
        raise TypeError("The decorated function must be an async function (coroutine).")

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            ctx = get_tool_context()
        except RuntimeError:
            # No cache context available, execute function directly
            return await func(*args, **kwargs)

        handler = _FunctionCacheHandler(func, ctx.cache_db)
        arg_hash = handler.compute_arg_hash(*args, **kwargs)

        cache_result = handler.lookup_cache(arg_hash)
        if cache_result.exists:
            logger.debug(f"Cache hit for {func.__name__}")
            return cache_result.value  # type: ignore

        logger.debug(f"Cache miss for {func.__name__}")
        actual_result = await func(*args, **kwargs)
        await asyncio.to_thread(handler.store_result, arg_hash, actual_result)
        return actual_result

    return wrapper

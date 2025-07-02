"""Function caching decorators.

This module provides decorators for caching function results in a database,
supporting both synchronous and asynchronous functions.
"""

import asyncio
import hashlib
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, ParamSpec, Protocol, TypeVar

from pydantic import BaseModel

from tidyllm.schema import FunctionDescription, parse_from_json


class DatabaseProtocol(Protocol):
    """Protocol defining the minimal database interface required for caching."""

    def mutate(self, sql: str, params: list[Any] | None = None) -> int:
        """Execute an INSERT/UPDATE/DELETE statement.

        Args:
            sql: SQL statement to execute
            params: Optional parameters for the SQL statement

        Returns:
            Number of affected rows
        """
        ...

    def query(self, sql: str, params: list[Any] | None = None) -> Any:
        """Execute a SELECT statement and return results.

        Args:
            sql: SQL query to execute
            params: Optional parameters for the SQL query

        Returns:
            Query results (implementation-specific format)
        """
        ...


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
    db: DatabaseProtocol
    table_name: str
    description: FunctionDescription

    def __init__(self, func: Callable[P, R], db: DatabaseProtocol):
        self.func = func
        self.db = db
        self.description = FunctionDescription(func)
        self.table_name = f"{self.description.name}_cache"

        self._ensure_cache_table()

    def _ensure_cache_table(self):
        """Ensure the cache table exists."""
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                arg_hash TEXT PRIMARY KEY,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.db.mutate(sql)

    def compute_arg_hash(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Compute a hash of the function arguments."""
        args_instance = self.description.arg_model_from_args(*args, **kwargs)
        args_json = args_instance.model_dump_json()
        args_hash = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        return args_hash

    def lookup_cache(self, arg_hash: str) -> CacheResult[R]:
        """Single method to check cache and return result if exists."""
        sql = f"SELECT result FROM {self.table_name} WHERE arg_hash = ?"
        cursor = self.db.query(sql, [arg_hash])

        row = cursor.first()
        if row:
            result_data = json.loads(row.result)
            parsed_result = parse_from_json(
                result_data["result"], self.description.result_type
            )
            return CacheResult.hit(parsed_result)

        return CacheResult.miss()

    def store_result(self, arg_hash: str, result_data: R):
        """Store result in cache."""
        if isinstance(result_data, BaseModel):
            result_data = result_data.model_dump() # type: ignore

        result = {"result": result_data}
        result_json = json.dumps(result)
        sql = (
            f"INSERT OR REPLACE INTO {self.table_name} (arg_hash, result) VALUES (?, ?)"
        )
        self.db.mutate(sql, [arg_hash, result_json])


def cached_function(db: DatabaseProtocol):
    """Decorator for caching synchronous function results.

    Args:
        db: Database instance implementing DatabaseProtocol

    Returns:
        Decorator function

    Example:
        >>> db = Database()
        >>> @cached_function(db)
        ... def expensive_computation(x: int) -> int:
        ...     return x * x
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        handler = _FunctionCacheHandler(func, db)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            arg_hash = handler.compute_arg_hash(*args, **kwargs)

            cache_result = handler.lookup_cache(arg_hash)
            if cache_result.exists:
                return cache_result.value  # type: ignore

            actual_result = func(*args, **kwargs)
            handler.store_result(arg_hash, actual_result)
            return actual_result

        return wrapper

    return decorator


def async_cached_function(db: DatabaseProtocol):
    """Decorator for caching asynchronous function results.

    Args:
        db: Database instance implementing DatabaseProtocol

    Returns:
        Decorator function

    Example:
        >>> db = Database()
        >>> @async_cached_function(db)
        ... async def fetch_data(url: str) -> dict:
        ...     # expensive async operation
        ...     return {"data": "result"}
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("The decorated function must be an async function (coroutine).")

        handler = _FunctionCacheHandler(func, db)

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            arg_hash = handler.compute_arg_hash(*args, **kwargs)

            cache_result = await asyncio.to_thread(handler.lookup_cache, arg_hash)
            if cache_result.exists:
                return cache_result.value  # type: ignore

            actual_result = await func(*args, **kwargs)
            await asyncio.to_thread(handler.store_result, arg_hash, actual_result)
            return actual_result

        return wrapper

    return decorator

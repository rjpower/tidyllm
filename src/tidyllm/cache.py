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

from tidyllm.context import get_tool_context
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


class CacheContextProtocol(Protocol):
    """Protocol for cache context that provides database access."""
    
    db: DatabaseProtocol


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
    cache_context: CacheContextProtocol
    table_name: str
    description: FunctionDescription

    def __init__(self, func: Callable[P, R], cache_context: CacheContextProtocol):
        self.func = func
        self.cache_context = cache_context
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
        self.cache_context.db.mutate(sql)

    def compute_arg_hash(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Compute a hash of the function arguments."""
        args_instance = self.description.arg_model_from_args(*args, **kwargs)
        args_json = args_instance.model_dump_json()
        args_hash = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        return args_hash

    def lookup_cache(self, arg_hash: str) -> CacheResult[R]:
        """Single method to check cache and return result if exists."""
        sql = f"SELECT result FROM {self.table_name} WHERE arg_hash = ?"
        cursor = self.cache_context.db.query(sql, [arg_hash])

        row = cursor.first()
        if row:
            result_data = json.loads(row.result)
            parsed_result = parse_from_json(result_data, self.description.result_type)
            return CacheResult.hit(parsed_result)

        return CacheResult.miss()

    def store_result(self, arg_hash: str, result_data: R):
        """Store result in cache."""
        if isinstance(result_data, BaseModel):
            result_json = result_data.model_dump_json()
        else:
            result_json = json.dumps(result_data)
        sql = (
            f"INSERT OR REPLACE INTO {self.table_name} (arg_hash, result) VALUES (?, ?)"
        )
        self.cache_context.db.mutate(sql, [arg_hash, result_json])


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
            cache_context = get_tool_context()
        except RuntimeError:
            # No cache context available, execute function directly
            return func(*args, **kwargs)
        
        if not hasattr(cache_context, 'db'):
            # Context doesn't have caching capability, execute function directly
            return func(*args, **kwargs)
            
        handler = _FunctionCacheHandler(func, cache_context)
        arg_hash = handler.compute_arg_hash(*args, **kwargs)
        
        cache_result = handler.lookup_cache(arg_hash)
        if cache_result.exists:
            return cache_result.value  # type: ignore

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
            cache_context = get_tool_context()
        except RuntimeError:
            # No cache context available, execute function directly
            return await func(*args, **kwargs)
        
        if not hasattr(cache_context, 'db'):
            # Context doesn't have caching capability, execute function directly
            return await func(*args, **kwargs)
            
        handler = _FunctionCacheHandler(func, cache_context)
        arg_hash = handler.compute_arg_hash(*args, **kwargs)

        cache_result = await asyncio.to_thread(handler.lookup_cache, arg_hash)
        if cache_result.exists:
            return cache_result.value  # type: ignore

        actual_result = await func(*args, **kwargs)
        await asyncio.to_thread(handler.store_result, arg_hash, actual_result)
        return actual_result

    return wrapper

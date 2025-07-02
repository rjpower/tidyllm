"""Function caching decorators.

This module provides decorators for caching function results in a database,
supporting both synchronous and asynchronous functions.
"""

import asyncio
import hashlib
import json
from collections.abc import Callable, Coroutine
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, cast

from pydantic import BaseModel

from tidyllm.schema import FunctionDescription, parse_from_json


class DatabaseProtocol(Protocol):
    """Protocol defining the minimal database interface required for caching."""

    def create_table(self, table_name: str, schema: dict[str, str]) -> None:
        """Create a table with the given schema.

        Args:
            table_name: Name of the table to create
            schema: Dictionary mapping column names to SQL type definitions
                   e.g., {"id": "INTEGER PRIMARY KEY", "data": "TEXT"}
        """
        ...

    def get(
        self, table_name: str, keys: dict[str, Any], key_columns: tuple[str, ...] = ("id",)
    ) -> dict[str, Any] | None:
        """Get a single row by key.

        Args:
            table_name: Name of the table to query
            keys: Dictionary mapping column names to values
            key_columns: Tuple of column names that form the key (default: ("id",))

        Returns:
            Dictionary of column values if found, None otherwise
        """
        ...

    def update(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        key_columns: tuple[str, ...] = ("id",),
    ) -> None:
        """Insert or update a row.

        Args:
            table_name: Name of the table to update
            data: Dictionary of column values to insert/update
            key_columns: Tuple of column names that form the key (default: ("id",))
        """
        ...


R = TypeVar("R")
P = ParamSpec("P")


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
        schema = {
            "arg_hash": "TEXT PRIMARY KEY",
            "result": "TEXT",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        }
        self.db.create_table(self.table_name, schema)

    def compute_arg_hash(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Compute a hash of the function arguments."""
        args_instance = self.description.arg_model_from_args(*args, **kwargs)
        args_json = args_instance.model_dump_json()
        args_hash = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        return args_hash

    def get_cached_result_sync(self, arg_hash: str) -> R | None:
        """Get cached result synchronously."""
        row = self.db.get(self.table_name, {"arg_hash": arg_hash}, key_columns=("arg_hash",))
        if row:
            return parse_from_json(row["result"], self.description.result_type)
        return None

    async def get_cached_result_async(self, arg_hash: str) -> R | None:
        """Get cached result asynchronously."""
        # Run synchronous DB operation in a thread pool
        row = await asyncio.to_thread(
            self.db.get, self.table_name, {"arg_hash": arg_hash}, key_columns=("arg_hash",)
        )
        if row:
            return parse_from_json(row["result"], self.description.result_type)
        return None

    def store_result_sync(self, arg_hash: str, result_data: R):
        """Store result synchronously."""
        if isinstance(result_data, BaseModel):
            result_data = result_data.model_dump() # type: ignore

        result = { "result": result_data }
        data = {"arg_hash": arg_hash, "result": json.dumps(result)}
        self.db.update(self.table_name, [data], key_columns=("arg_hash",))

    async def store_result_async(self, arg_hash: str, result_data: R):
        """Store result asynchronously."""
        if isinstance(result_data, BaseModel):
            result_data = result_data.model_dump() # type: ignore

        result = { "result": result_data }
        data = {"arg_hash": arg_hash, "result": json.dumps(result)}
        await asyncio.to_thread(self.db.update, self.table_name, [data], key_columns=("arg_hash",))


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

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            arg_hash = handler.compute_arg_hash(*args, **kwargs)

            cached_result = handler.get_cached_result_sync(arg_hash)
            if cached_result is not None:
                return cached_result

            actual_result = func(*args, **kwargs)
            handler.store_result_sync(arg_hash, actual_result)
            return actual_result

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__annotations__ = func.__annotations__

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

        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            arg_hash = handler.compute_arg_hash(*args, **kwargs)

            cached_result = await handler.get_cached_result_async(arg_hash)
            if cached_result is not None:
                return cast(R, cached_result)

            actual_result = await func(*args, **kwargs)
            await handler.store_result_async(arg_hash, actual_result)
            return actual_result

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator

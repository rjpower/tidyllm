"""Utilities to import Python files from a particular package or directory for automatic registration."""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_module_name_from_path(file_path: Path, base_path: Path) -> str:
    """Generate a module name from a file path relative to base path."""
    relative_path = file_path.relative_to(base_path)
    # Remove .py extension and convert path separators to dots
    module_parts = relative_path.with_suffix("").parts
    return ".".join(module_parts)


def _import_module_from_path(file_path: Path, module_name: str) -> None:
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    # # Add to sys.modules with the full qualified name for proper pickling
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def discover_tools_in_directory(
    directory: Path,
    pattern: str = "*.py",
    recursive: bool = True,
    exclude_patterns: list[str] | None = None,
    package_prefix: str | None = None,
):
    """
    Auto-discover tools in a directory by importing Python modules.

    Args:
        directory: Directory to search for tools
        pattern: File pattern to match (default: "*.py")
        recursive: Whether to search recursively
        exclude_patterns: Patterns to exclude (e.g., ["test_*", "*_test.py"])
        package_prefix: Optional package prefix to add to module names (e.g., "tidyllm.tools")

    Returns:
        List of discovered FunctionDescription objects from the specified directory

    Example:
        # Discover all tools in a directory
        tools = discover_tools_in_directory(
            Path("my_tools"),
            exclude_patterns=["test_*", "__pycache__"]
        )
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "test_*.py",
            "__pycache__",
            "*.pyc",
            "tests/",
            "__main__.py",
            "__init__.py",
        ]

    # Find Python files to discover
    if recursive:
        python_files = directory.rglob(pattern)
    else:
        python_files = directory.glob(pattern)

    # Convert to absolute paths for comparison
    discovered_file_paths = set()
    for py_file in python_files:
        # Skip files matching exclude patterns
        if any(py_file.match(exclude_pattern) for exclude_pattern in exclude_patterns):
            logger.debug(f"Skipping excluded file: {py_file}")
            continue

        logger.debug(f"Inspecting {py_file} for registered functions.")
        discovered_file_paths.add(py_file.resolve())

        relative_module_name = _get_module_name_from_path(py_file, directory)
        if package_prefix:
            module_name = f"{package_prefix}.{relative_module_name}"
        else:
            module_name = relative_module_name

        try:
            _import_module_from_path(py_file, module_name)
        except Exception as e:
            logger.warning(
                f"Failed to import {module_name} from {py_file} during discovery: {e}"
            )

    logger.info(f"Auto-discovery completed. Found {len(discovered_file_paths)} files")


def discover_tools_in_package(package_name: str, recursive: bool = True):
    """
    Auto-discover tools in a Python package.

    Args:
        package_name: Name of the package to search (e.g., "tinyagent.tools")
        recursive: Whether to search subpackages

    Returns:
        List of discovered FunctionDescription objects

    Example:
        # Discover all tools in the tinyagent.tools package
        tools = discover_tools_in_package("tinyagent.tools")
    """
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__") or not package.__path__:
        logger.warning(f"{package_name} is not a package or has no __path__")
        return []

    package_path = Path(list(package.__path__)[0])
    return discover_tools_in_directory(
        package_path, recursive=recursive, package_prefix=package_name
    )

"""Docstring parsing using griffe for enhanced parameter extraction."""

import logging
from collections.abc import Callable
from typing import Any

from griffe import Docstring
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocstringInfo(BaseModel):
    """Structured docstring information extracted from function documentation."""
    
    description: str
    returns: str
    parameters: dict[str, str]


def extract_docs_from_string(docstring_text: str) -> DocstringInfo:
    """Extract documentation including parameters from a docstring using griffe.

    Args:
        docstring_text: The docstring text to parse

    Returns:
        DocstringInfo model containing description, returns, and parameter docs
    """
    if not docstring_text:
        return DocstringInfo(description="", returns="", parameters={})

    # Parse docstring directly using griffe
    docstring = Docstring(docstring_text, lineno=1)
    parsed = docstring.parse("google", warnings=False)

    description = ""
    returns = ""
    parameters: dict[str, str] = {}

    # Extract from parsed sections
    for section in parsed:
        if section.kind.value == "text":
            # This is the description/summary section
            if hasattr(section, "value") and section.value:
                description = section.value

        elif section.kind.value == "returns":
            if hasattr(section, "value") and section.value:
                for ret in section.value:
                    if hasattr(ret, "description"):
                        returns = ret.description
                        break

        elif section.kind.value == "parameters":
            if hasattr(section, "value") and section.value:
                for param in section.value:
                    if hasattr(param, "name") and hasattr(param, "description"):
                        parameters[param.name] = param.description

    return DocstringInfo(
        description=description,
        returns=returns,
        parameters=parameters
    )


def update_schema_with_docstring(
    schema: dict[str, Any], docstring_text: str | None = None
) -> dict[str, Any]:
    """Enhance existing schema with griffe-extracted documentation.

    Args:
        schema: Existing tool schema to enhance
        docstring_text: Optional docstring text to parse for documentation

    Returns:
        Enhanced schema with better documentation
    """
    if not docstring_text:
        return schema

    func_docs = extract_docs_from_string(docstring_text)

    # Enhance function description if not already set
    if func_docs.description and not schema.get("function", {}).get("description"):
        schema.setdefault("function", {})["description"] = func_docs.description

    # Enhance parameter descriptions
    if "function" in schema and "parameters" in schema["function"]:
        parameters = schema["function"]["parameters"]
        if "properties" in parameters:
            for param_name, param_schema in parameters["properties"].items():
                if param_name in func_docs.parameters and "description" not in param_schema:
                    param_schema["description"] = func_docs.parameters[param_name]

    return schema

"""FastAPI adapter for TidyLLM registry functions."""

from typing import Any

from fastapi import FastAPI, HTTPException

from tidyllm.library import FunctionLibrary
from tidyllm.models import ToolError
from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext


def create_fastapi_app(
    context: ToolContext | None = None,
    title: str = "TidyLLM Tools API",
    description: str = "API for TidyLLM registered tools",
    version: str = "1.0.0",
) -> FastAPI:
    """
    Create a FastAPI application that exposes TidyLLM tools as individual endpoints.

    Each tool gets its own POST endpoint that accepts the tool's specific argument types
    directly, with FastAPI automatically generating the OpenAPI schema.

    Args:
        context: Optional ToolContext for tools execution
        title: API title
        description: API description
        version: API version

    Returns:
        Configured FastAPI application with individual tool endpoints

    Example:
        from tidyllm.adapters.fastapi_adapter import create_fastapi_app
        from tidyllm.tools.context import ToolContext

        # Create with context
        context = ToolContext()
        app = create_fastapi_app(context)

        # Run with: uvicorn module:app --reload
        # Tools will be available at /tools/{tool_name}
    """
    app = FastAPI(title=title, description=description, version=version)

    # Use provided context or create default
    if context is None:
        from tidyllm.tools.config import Config
        context = ToolContext(config=Config())
    
    # Store context for use in endpoints
    app._tidyllm_context = context

    # Get tools from registry
    from tidyllm.registry import REGISTRY
    function_descriptions = REGISTRY.functions

    @app.get("/", summary="API Information")
    async def root():
        """Get basic API information."""
        tool_names = [desc.name for desc in function_descriptions]
        return {
            "title": title,
            "description": description,
            "version": version,
            "tools": tool_names,
            "endpoints": {f"/tools/{name}" for name in tool_names}
        }

    @app.get("/health", summary="Health Check")
    async def health_check():
        """Simple health check endpoint."""
        return {
            "status": "healthy",
            "available_tools": len(function_descriptions)
        }

    # Create individual endpoints for each tool
    for tool_desc in function_descriptions:
        _create_tool_endpoint(app, tool_desc)

    return app


def _create_tool_endpoint(app: FastAPI, tool_desc):
    """Create a FastAPI endpoint for a specific tool.""" 
    from inspect import Parameter, Signature
    
    tool_name = tool_desc.name
    args_model = tool_desc.args_model
    
    # Create the endpoint function dynamically
    def create_endpoint():
        async def tool_endpoint(args):
            """Execute the tool with the provided arguments."""
            try:
                # Set context for this request
                with set_tool_context(app._tidyllm_context):
                    # Validate and execute tool directly
                    parsed_args = tool_desc.validate_and_parse_args(args.model_dump())
                    result = tool_desc.call(**parsed_args)
                    return result

            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                # Handle unexpected errors
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error: {str(e)}"
                ) from e

        # Set function metadata
        tool_endpoint.__name__ = f"{tool_name}_endpoint"
        tool_endpoint.__doc__ = tool_desc.function.__doc__ or f"Execute {tool_name} tool"
        
        # Create signature with properly typed parameter
        param = Parameter('args', Parameter.POSITIONAL_OR_KEYWORD, annotation=args_model)
        tool_endpoint.__signature__ = Signature([param])
        
        return tool_endpoint
    
    endpoint_func = create_endpoint()

    # Add the endpoint to the FastAPI app
    app.post(
        f"/tools/{tool_name}",
        response_model=None,  # Let FastAPI infer from return type
        summary=f"Execute {tool_name}",
        description=tool_desc.function.__doc__ or f"Execute the {tool_name} tool",
        tags=["tools"]
    )(endpoint_func)


# Convenience function for common use case
def create_portkit_api(
    context: ToolContext | None = None,
    title: str = "PortKit Tools API",
    description: str = "API for PortKit TinyAgent tools",
) -> FastAPI:
    """
    Create a FastAPI app specifically for PortKit tools.

    Args:
        context: ToolContext for tools execution
        title: API title
        description: API description

    Returns:
        Configured FastAPI application with PortKit tools

    Example:
        from tidyllm.adapters.fastapi_adapter import create_portkit_api
        from tidyllm.tools.context import ToolContext

        context = ToolContext()
        app = create_portkit_api(context=context)
    """
    # Import all tools to ensure they're registered
    import tidyllm.tools.anki  # noqa: F401
    import tidyllm.tools.manage_db  # noqa: F401
    import tidyllm.tools.notes  # noqa: F401
    import tidyllm.tools.transcribe  # noqa: F401
    import tidyllm.tools.vocab_table  # noqa: F401

    return create_fastapi_app(context=context, title=title, description=description)

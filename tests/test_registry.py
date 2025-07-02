"""Tests for tool registry system."""

from pydantic import BaseModel

from tidyllm.registry import REGISTRY, Registry, ToolError


class RegistryTestArgs(BaseModel):
    """Test arguments model."""

    message: str
    count: int = 1


class RegistryTestResult(BaseModel):
    """Test result model."""

    output: str
    processed_count: int


def registry_test_function(args: RegistryTestArgs) -> RegistryTestResult:
    """A test function for registry."""
    return RegistryTestResult(output=f"Processed: {args.message}", processed_count=args.count)


def another_registry_test_function(args: RegistryTestArgs) -> dict:
    """Another test function."""
    return {"message": args.message, "count": args.count}


def failing_test_function(args: RegistryTestArgs) -> RegistryTestResult:
    """A test function that always fails."""
    raise RuntimeError("This function always fails")


def multi_param_function(name: str, count: int, enabled: bool = True) -> dict:
    """Function with multiple parameters."""
    return {"name": name, "count": count, "enabled": enabled}


def primitive_param_function(message: str) -> dict:
    """Function with single primitive parameter."""
    return {"processed": message.upper()}


class TestRegistry:
    """Test Registry class."""

    def setup_method(self):
        """Set up test registry."""
        self.registry = Registry()

    def test_registry_initialization(self):
        """Test registry initializes empty."""
        assert len(self.registry._tools) == 0
        assert len(self.registry.functions) == 0

    def test_register_function(self):
        """Test registering a function."""
        self.registry.register(registry_test_function)

        assert "registry_test_function" in self.registry._tools
        assert self.registry.get_description("registry_test_function") is not None

    def test_register_duplicate_function(self):
        """Test that registering duplicate function name is silently ignored."""
        self.registry.register(registry_test_function)
        assert len(self.registry.functions) == 1

        # Registering again should be silently ignored
        self.registry.register(registry_test_function)
        assert len(self.registry.functions) == 1

    def test_register_with_context_type(self):
        """Test registering function works properly."""

        # Test that registration works for functions regardless of context needs
        self.registry.register(registry_test_function)

        func_desc = self.registry.get_description("registry_test_function")
        assert func_desc is not None
        # Verify basic function properties are preserved
        assert func_desc.name == "registry_test_function"
        assert func_desc.function is registry_test_function

    def test_get_existing_tool(self):
        """Test getting an existing tool."""
        self.registry.register(registry_test_function)

        func_desc = self.registry.get_description("registry_test_function")
        assert func_desc is not None
        assert func_desc.function is registry_test_function

        # Test get_function method
        tool_func = self.registry.get_function("registry_test_function")
        assert tool_func is registry_test_function

    def test_get_nonexistent_tool(self):
        """Test getting a non-existent tool."""
        func_desc = self.registry.get_description("nonexistent")
        assert func_desc is None

    def test_list_tools(self):
        """Test listing all registered tools."""
        self.registry.register(registry_test_function)
        self.registry.register(another_registry_test_function)

        functions = self.registry.functions
        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "registry_test_function" in names
        assert "another_registry_test_function" in names

    def test_registry_independence(self):
        """Test that separate registry instances are independent."""
        other_registry = Registry()

        self.registry.register(registry_test_function)
        assert len(self.registry.functions) == 1
        assert len(other_registry.functions) == 0

        other_registry.register(another_registry_test_function)
        assert len(self.registry.functions) == 1
        assert len(other_registry.functions) == 1

    def test_function_metadata_attached(self):
        """Test that metadata is attached to registered functions."""
        self.registry.register(registry_test_function)

        # Check function metadata via registry
        func_desc = self.registry.get_description("registry_test_function")
        assert func_desc is not None
        assert func_desc.function_schema["function"]["name"] == "registry_test_function"
        # Check FunctionDescription has schema
        assert func_desc.function_schema is not None


class TestGlobalRegistry:
    """Test global REGISTRY instance."""

    def setup_method(self):
        """Save global registry state before each test."""
        self._saved_tools = REGISTRY._tools.copy()

    def teardown_method(self):
        """Restore global registry state after each test."""
        REGISTRY._tools = self._saved_tools

    def test_global_registry_exists(self):
        """Test that global REGISTRY exists."""
        assert REGISTRY is not None
        assert isinstance(REGISTRY, Registry)

    def test_global_registry_register(self):
        """Test registering with global registry."""
        REGISTRY.register(registry_test_function)

        names = [f.name for f in REGISTRY.functions]
        assert "registry_test_function" in names

    def test_global_registry_isolation(self):
        """Test that tests don't interfere with each other."""
        # This test should start with the saved registry state
        initial_count = len(self._saved_tools)
        assert len(REGISTRY.functions) == initial_count

        REGISTRY.register(registry_test_function)

        assert len(REGISTRY.functions) == initial_count + 1


class TestRegistryExecution:
    """Test Registry execution capabilities."""

    def setup_method(self):
        """Set up test registry with execution functions."""
        self.registry = Registry()
        self.registry.register(registry_test_function)
        self.registry.register(another_registry_test_function)
        self.registry.register(failing_test_function)
        self.registry.register(multi_param_function)
        self.registry.register(primitive_param_function)

    def test_call_with_dict_arguments(self):
        """Test calling function with dictionary arguments."""
        result = self.registry.call("registry_test_function", {"message": "test", "count": 5})
        
        assert isinstance(result, RegistryTestResult)
        assert result.output == "Processed: test"
        assert result.processed_count == 5

    def test_call_with_default_arguments(self):
        """Test calling function with default arguments."""
        result = self.registry.call("registry_test_function", {"message": "default"})
        
        assert isinstance(result, RegistryTestResult)
        assert result.output == "Processed: default"
        assert result.processed_count == 1  # Default value

    def test_call_nonexistent_function(self):
        """Test calling a non-existent function returns ToolError."""
        result = self.registry.call("nonexistent", {})
        
        assert isinstance(result, ToolError)
        assert "not found" in result.error

    def test_call_with_invalid_arguments(self):
        """Test calling function with invalid arguments returns ToolError."""
        # Missing required argument
        result = self.registry.call("registry_test_function", {"count": 5})
        
        assert isinstance(result, ToolError)
        assert "Invalid arguments" in result.error

    def test_call_function_that_fails(self):
        """Test calling function that raises exception returns ToolError."""
        result = self.registry.call("failing_test_function", {"message": "test", "count": 1})
        
        assert isinstance(result, ToolError)
        assert "Tool execution failed" in result.error
        assert "always fails" in result.error

    def test_call_multi_param_function(self):
        """Test calling function with multiple parameters."""
        result = self.registry.call("multi_param_function", {
            "name": "test", 
            "count": 42, 
            "enabled": False
        })
        
        assert result == {"name": "test", "count": 42, "enabled": False}

    def test_call_multi_param_function_with_defaults(self):
        """Test calling function with default parameter values."""
        result = self.registry.call("multi_param_function", {
            "name": "test", 
            "count": 42
        })
        
        assert result == {"name": "test", "count": 42, "enabled": True}

    def test_call_primitive_param_function(self):
        """Test calling function with single primitive parameter."""
        result = self.registry.call("primitive_param_function", {"message": "hello"})
        
        assert result == {"processed": "HELLO"}

    def test_get_schemas(self):
        """Test getting OpenAI-format schemas for all tools."""
        schemas = self.registry.get_schemas()
        
        assert len(schemas) == 5
        schema_names = [s["function"]["name"] for s in schemas]
        assert "registry_test_function" in schema_names
        assert "multi_param_function" in schema_names
        assert "primitive_param_function" in schema_names

    def test_call_with_json_response(self):
        """Test call_with_json_response method."""
        result = self.registry.call_with_json_response(
            "registry_test_function", 
            {"message": "test", "count": 5},
            "test_id"
        )
        
        # Should return JSON string
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert parsed["output"] == "Processed: test"
        assert parsed["processed_count"] == 5

    def test_call_with_json_response_error(self):
        """Test call_with_json_response with error."""
        result = self.registry.call_with_json_response(
            "nonexistent", 
            {},
            "test_id"
        )
        
        # Should return JSON error string
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert "error" in parsed

    def test_add_tool_method(self):
        """Test add_tool method for dynamically adding tools."""
        def dynamic_tool(args: RegistryTestArgs) -> dict:
            return {"dynamic": True, "message": args.message}
        
        self.registry.add_tool("dynamic_tool", dynamic_tool)
        
        # Test it was added
        func_desc = self.registry.get_description("dynamic_tool")
        assert func_desc is not None
        assert func_desc.name == "dynamic_tool"
        
        # Test it can be called
        result = self.registry.call("dynamic_tool", {"message": "test"})
        assert result == {"dynamic": True, "message": "test"}

    def test_add_tool_duplicate_error(self):
        """Test add_tool raises error for duplicate tool names."""
        def duplicate_tool(args: RegistryTestArgs) -> dict:
            return {"duplicate": True}
        
        # First addition should work
        self.registry.add_tool("duplicate_tool", duplicate_tool)
        
        # Second addition should raise error
        try:
            self.registry.add_tool("duplicate_tool", duplicate_tool)
            assert False, "Expected ValueError for duplicate tool"
        except ValueError as e:
            assert "already exists" in str(e)

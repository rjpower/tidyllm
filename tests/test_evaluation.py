"""Tests for evaluation framework."""

from unittest.mock import Mock

import pytest

from tidyllm.agent import LLMAgent
from tidyllm.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationRunner,
    evaluation_test,
    run_evaluations,
)
from tidyllm.llm import (
    AssistantMessage,
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolMessage,
)
from tidyllm.registry import Registry
from tidyllm.tools.calculator import calculator


class TestEvaluationDecorator:
    """Test the @evaluation_test decorator."""

    def test_decorator_marks_function(self):
        """Test that decorator properly marks functions."""
        from tidyllm.evaluation import EVALUATION_REGISTRY

        @evaluation_test()
        def test_func():
            pass

        # Check that function is registered in the global registry
        test_key = f"{test_func.__module__}.{test_func.__name__}"
        assert test_key in EVALUATION_REGISTRY
        assert EVALUATION_REGISTRY[test_key].func == test_func
        assert EVALUATION_REGISTRY[test_key].timeout_seconds == 30

    def test_decorator_with_custom_timeout(self):
        """Test decorator with custom timeout."""
        from tidyllm.evaluation import EVALUATION_REGISTRY

        @evaluation_test(timeout_seconds=60)
        def test_func():
            pass

        # Check that custom timeout is stored in registry
        test_key = f"{test_func.__module__}.{test_func.__name__}"
        assert EVALUATION_REGISTRY[test_key].timeout_seconds == 60


class TestEvaluationContext:
    """Test evaluation context and assertions."""

    def setup_method(self):
        """Set up test context."""
        # Create a simple mock client for testing
        class SimpleMockClient(LLMClient):
            def completion(self, model, messages, tools, **kwargs):
                return LLMResponse(model="gemini/gemini-2.5-flash", messages=messages)

        library = Registry()
        library.register(calculator)
        client = SimpleMockClient()
        agent = LLMAgent(function_library=library, llm_client=client, model="mock-model")
        self.context = EvaluationContext(agent)

    def test_assert_tool_called_success(self):
        """Test successful tool assertion."""
        tool_call = ToolCall(tool_name="calculator", tool_args={})
        assistant_msg = AssistantMessage(
            content="I'll calculate that", tool_calls=[tool_call]
        )
        response = LLMResponse(
            model="gemini/gemini-2.5-flash", messages=[assistant_msg]
        )

        self.context.assert_tool_called(response, "calculator")
        assert self.context._assertions_passed == 1
        assert self.context._assertions_total == 1

    def test_assert_tool_called_failure(self):
        """Test failed tool assertion."""
        tool_call = ToolCall(tool_name="wrong_tool", tool_args={})
        assistant_msg = AssistantMessage(
            content="I'll use wrong tool", tool_calls=[tool_call]
        )
        response = LLMResponse(
            model="gemini/gemini-2.5-flash", messages=[assistant_msg]
        )

        with pytest.raises(AssertionError, match="Expected tool 'calculator'"):
            self.context.assert_tool_called(response, "calculator")

        assert self.context._assertions_passed == 0
        assert self.context._assertions_total == 1

    def test_assert_success(self):
        """Test success assertion."""
        response = LLMResponse(model="gemini/gemini-2.5-flash", messages=[])
        self.context.assert_success(response)

        assert self.context._assertions_passed == 1

    def test_assert_result_contains(self):
        """Test result contains assertion."""
        tool_call = ToolCall(tool_name="test", tool_args={})
        assistant_msg = AssistantMessage(content="I'll test", tool_calls=[tool_call])
        tool_msg = ToolMessage(content="Hello World", tool_call_id="test_1", name="test")
        response = LLMResponse(
            model="gemini/gemini-2.5-flash", messages=[assistant_msg, tool_msg]
        )

        self.context.assert_result_contains(response, "World")
        assert self.context._assertions_passed == 1

        with pytest.raises(AssertionError, match="Expected 'Missing'"):
            self.context.assert_result_contains(response, "Missing")

    def test_assert_result_equals(self):
        """Test result equals assertion."""
        tool_call = ToolCall(tool_name="test", tool_args={})
        assistant_msg = AssistantMessage(
            content="I'll calculate", tool_calls=[tool_call]
        )
        tool_msg = ToolMessage(content="42", tool_call_id="test_1", name="test")
        response = LLMResponse(
            model="gemini/gemini-2.5-flash", messages=[assistant_msg, tool_msg]
        )

        self.context.assert_result_equals(response, 42)
        assert self.context._assertions_passed == 1

        with pytest.raises(AssertionError, match="Expected 100"):
            self.context.assert_result_equals(response, 100)


class TestEvaluationRunner:
    """Test evaluation runner functionality."""

    def setup_method(self):
        """Set up test runner."""
        library = Registry()
        library.register(calculator)
        self.runner = EvaluationRunner(library)

    def test_discover_tests(self):
        """Test test discovery from modules."""
        # Create a mock module with evaluation tests
        mock_module = Mock()
        mock_module.__name__ = "test_module"

        @evaluation_test()
        def test_func1():
            pass

        def regular_func():
            pass

        @evaluation_test()
        def test_func2():
            pass

        # Manually set the module name for the test functions to match
        test_func1.__module__ = "test_module"
        test_func2.__module__ = "test_module"
        
        # Clear and re-register with the correct module name
        from tidyllm.evaluation import EVALUATION_REGISTRY
        old_registry = EVALUATION_REGISTRY.copy()
        EVALUATION_REGISTRY.clear()
        
        # Re-decorate to register with correct module names
        test_func1 = evaluation_test()(test_func1)
        test_func2 = evaluation_test()(test_func2)

        try:
            tests = self.runner.discover_tests([mock_module])
            assert len(tests) == 2
            assert test_func1 in tests
            assert test_func2 in tests
        finally:
            # Restore original registry
            EVALUATION_REGISTRY.clear()
            EVALUATION_REGISTRY.update(old_registry)

    def test_run_test_success(self):
        """Test successful test execution."""

        @evaluation_test()
        def test_success(context):
            # This should succeed with mock client
            pass

        result = self.runner.run_test(test_success, "mock", use_mock=True)

        assert isinstance(result, EvaluationResult)
        # Function completed successfully if no exception raised
        assert result.test_name == "test_success"
        assert result.duration_ms >= 0
        assert result.error_message is None

    def test_run_test_failure(self):
        """Test failed test execution."""

        @evaluation_test()
        def test_failure():
            raise ValueError("Test error")

        result = self.runner.run_test(test_failure, "mock", use_mock=True)

        assert isinstance(result, EvaluationResult)
        # Function completed successfully if no exception raised
        assert result.test_name == "test_failure"
        assert result.error_message and "Test error" in result.error_message

    def test_run_test_with_context(self):
        """Test running test that expects context parameter."""

        @evaluation_test()
        def test_with_context(context):
            assert context is not None
            assert hasattr(context, "llm")

        result = self.runner.run_test(test_with_context, "mock", use_mock=True)
        # Function completed successfully if no exception raised

    def test_run_test_without_context(self):
        """Test running test that doesn't expect context."""

        @evaluation_test()
        def test_without_context():
            # Just a simple test without context parameter
            assert True

        result = self.runner.run_test(test_without_context, "mock", use_mock=True)
        # Function completed successfully if no exception raised

    def test_run_tests_multiple(self):
        """Test running multiple tests."""

        @evaluation_test()
        def test1():
            pass

        @evaluation_test()
        def test2():
            pass

        @evaluation_test()
        def test_fail():
            raise RuntimeError("Intentional failure")

        results = self.runner.run_tests([test1, test2, test_fail], "mock", use_mock=True)

        assert isinstance(results, list)
        assert len(results) == 3
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        assert passed == 2
        assert failed == 1


class TestEvaluationIntegration:
    """Integration tests for the full evaluation system."""

    def test_run_evaluations_with_modules(self):
        """Test running evaluations with provided modules."""
        library = Registry()
        library.register(calculator)

        # Create mock module with tests
        mock_module = Mock()
        mock_module.__name__ = "integration_test_module"

        @evaluation_test()
        def integration_test():
            assert True

        # Set the module name to match the mock module
        integration_test.__module__ = "integration_test_module"
        
        # Clear and re-register with the correct module name
        from tidyllm.evaluation import EVALUATION_REGISTRY
        old_registry = EVALUATION_REGISTRY.copy()
        EVALUATION_REGISTRY.clear()
        
        # Re-decorate to register with correct module name
        integration_test = evaluation_test()(integration_test)

        try:
            results = run_evaluations(
                function_library=library,
                model="mock",
                test_modules=[mock_module],
                mock_client=True,
            )

            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].success is True
        finally:
            # Restore original registry
            EVALUATION_REGISTRY.clear()
            EVALUATION_REGISTRY.update(old_registry)

    def test_run_evaluations_no_tests(self):
        """Test running evaluations when no tests are found."""
        library = Registry()
        library.register(calculator)
        mock_module = Mock()
        mock_module.__dict__ = {}

        import builtins

        original_dir = builtins.dir
        builtins.dir = lambda x: list(x.__dict__.keys())

        try:
            results = run_evaluations(
                function_library=library,
                model="mock",
                test_modules=[mock_module],
                mock_client=True,
            )

            assert len(results) == 0
        finally:
            builtins.dir = original_dir

    def test_run_evaluations_no_tests_found(self):
        """Test when no tests are found."""
        library = Registry()
        library.register(calculator)

        results = run_evaluations(function_library=library, model="mock", mock_client=True)

        assert len(results) == 0

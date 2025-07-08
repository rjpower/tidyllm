"""Benchmark framework for testing TidyAgent tools with LLMs."""

import inspect
import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, cast, overload

import click

from tidyllm.agent import LLMAgent
from tidyllm.llm import AssistantMessage, LiteLLMClient, LLMResponse, Role
from tidyllm.registry import Registry

P = ParamSpec("P")
T = TypeVar("T", covariant=True)


@dataclass
class EvaluationTestInfo:
    """Information about a registered evaluation test."""
    
    func: Callable
    timeout_seconds: int = 30
    module_name: str = ""


# Global registry for evaluation tests
EVALUATION_REGISTRY: dict[str, EvaluationTestInfo] = {}


@dataclass
class EvaluationResult:
    """Result of running a single evaluation test."""

    test_name: str
    success: bool
    duration_ms: int
    llm_response: LLMResponse | None = None
    error_message: str | None = None
    assertions_passed: int = 0
    assertions_total: int = 0


@overload
def evaluation_test(
    func_or_timeout: Callable[P, T],
) -> Callable[P, T]: ...


@overload
def evaluation_test(
    func_or_timeout: None = None,
    *,
    timeout_seconds: int = 30,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def evaluation_test(
    func_or_timeout: Callable[P, T] | None = None,
    *,
    timeout_seconds: int = 30,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to mark a function as a evaluation test.

    Can be used with or without parentheses:
        @evaluation_test
        def my_test(...): ...

        @evaluation_test()
        def my_test(...): ...

        @evaluation_test(timeout_seconds=60)
        def my_test(...): ...

    Args:
        func_or_timeout: Function (when used without parentheses)
        timeout_seconds: Timeout in seconds for the test
    """

    def _mark_evaluation_test(
        func: Callable[P, T], timeout: int = 30
    ) -> Callable[P, T]:
        # Get module name for namespacing
        module_name = getattr(func, "__module__", "")
        test_key = f"{module_name}.{func.__name__}"
        
        # Register in global registry
        EVALUATION_REGISTRY[test_key] = EvaluationTestInfo(
            func=func,
            timeout_seconds=timeout,
            module_name=module_name
        )
        return func

    # If first argument is a callable, this is direct usage (@evaluation_test)
    if callable(func_or_timeout):
        return _mark_evaluation_test(func_or_timeout, timeout_seconds)

    # Otherwise, this is parameterized usage (@evaluation_test() or @evaluation_test(timeout_seconds=60))
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return _mark_evaluation_test(func, timeout_seconds)

    return decorator


def find_test_cases(module):
    """Find all functions marked with @evaluation_test in a module."""
    test_cases = []
    module_name = getattr(module, "__name__", "")
    
    # Look for tests registered for this module
    for test_info in EVALUATION_REGISTRY.values():
        if test_info.module_name == module_name:
            test_cases.append(test_info.func)
    
    return test_cases


class EvaluationContext:
    """Context object provided to evaluation tests."""

    def __init__(self, llm: LLMAgent):
        self.llm = llm
        self._assertions_passed = 0
        self._assertions_total = 0
        self._test_name = ""

    def assert_tool_called(self, response: LLMResponse, expected_tool: str):
        """Assert that the expected tool was called."""
        self._assertions_total += 1

        # Collect all tool calls from assistant messages
        all_tool_calls = []
        for msg in response.messages:
            if msg.role == Role.ASSISTANT:
                all_tool_calls.extend(cast(AssistantMessage, msg).tool_calls)

        if any(tool_call.tool_name == expected_tool for tool_call in all_tool_calls):
            self._assertions_passed += 1
        else:
            raise AssertionError(
                f"Expected tool '{expected_tool}', but got '{[tool_call.tool_name for tool_call in all_tool_calls]}'"
            )

    def assert_success(self, response: LLMResponse):
        """Assert that the LLM response was successful."""
        self._assertions_total += 1
        self._assertions_passed += 1

    def assert_result_contains(self, response: LLMResponse, expected_value: Any):
        """Assert that the tool result contains the expected value."""
        self._assertions_total += 1

        # Collect all tool responses from tool messages
        tool_responses = []
        for msg in response.messages:
            if msg.role.value == "tool":
                tool_responses.append(msg.content)

        if any(expected_value in str(response) for response in tool_responses):
            self._assertions_passed += 1
        else:
            raise AssertionError(
                f"Expected '{expected_value}' in tool responses, got: {tool_responses}"
            )

    def assert_result_equals(self, response: LLMResponse, expected_value: Any):
        """Assert that the tool result equals the expected value."""
        self._assertions_total += 1

        # Collect all tool responses from tool messages
        tool_responses = []
        for msg in response.messages:
            if msg.role.value == "tool":
                try:
                    parsed = json.loads(msg.content)
                    if isinstance(parsed, dict) and "result" in parsed:
                        tool_responses.append(parsed["result"])
                    else:
                        tool_responses.append(parsed)
                except json.JSONDecodeError:
                    # If not JSON, use the content directly
                    tool_responses.append(msg.content)

        if any(response == expected_value for response in tool_responses):
            self._assertions_passed += 1
        else:
            raise AssertionError(
                f"Expected {expected_value}, got: {tool_responses}"
            )


class EvaluationRunner:
    """Runner for executing evaluation tests."""

    def __init__(self, function_library: Registry = None, test_cases: list[Callable] = None):
        self.function_library = function_library
        self.test_cases = test_cases or []

    def discover_tests(self, test_modules: list[Any]) -> list[Callable]:
        """Discover evaluation tests in the provided modules.

        Args:
            test_modules: List of Python modules containing evaluation tests

        Returns:
            List of test functions marked with @evaluation_test
        """
        tests = []

        for module in test_modules:
            module_name = getattr(module, "__name__", "")
            # Look for tests registered for this module
            for test_info in EVALUATION_REGISTRY.values():
                if test_info.module_name == module_name:
                    tests.append(test_info.func)

        return tests

    def run_test(self, test_func: Callable, model: str) -> EvaluationResult:
        """Run a single evaluation test.

        Args:
            test_func: Test function to execute
            model: LLM model to use for testing

        Returns:
            EvaluationResult with test execution details
        """
        start_time = time.time()
        test_name = getattr(test_func, "__name__", str(test_func))

        try:
            llm_client = LiteLLMClient()

            llm_agent = LLMAgent(
                model=model,
                function_library=self.function_library,
                llm_client=llm_client,
            )

            # Create test context
            context = EvaluationContext(llm_agent)
            context._test_name = test_name

            # Execute the test function
            # Check if test function expects context parameter
            sig = inspect.signature(test_func)
            if len(sig.parameters) > 0:
                test_func(context)
            else:
                test_func()

            duration_ms = int((time.time() - start_time) * 1000)

            return EvaluationResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                assertions_passed=context._assertions_passed,
                assertions_total=context._assertions_total,
            )

        except Exception as e:
            traceback.print_exc()
            duration_ms = int((time.time() - start_time) * 1000)

            return EvaluationResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e),
                assertions_passed=0,
                assertions_total=0,
            )

    def run_tests(
        self,
        tests: list[Callable],
        model: str,
    ) -> list[EvaluationResult]:
        """Run multiple evaluation tests and collect results.

        Args:
            tests: List of test functions to execute
            model: LLM model to use for testing

        Returns:
            List of EvaluationResult objects
        """
        results = []

        for test_func in tests:
            result = self.run_test(test_func, model)
            results.append(result)

            # Print progress
            status = "PASS" if result.success else "FAIL"
            print(f"{status}: {result.test_name} ({result.duration_ms}ms)")
            if not result.success:
                print(f"  Error: {result.error_message}")

        return results

    def run_tests_parallel(
        self, tests: list[Callable], model: str, max_workers: int = None
    ) -> list[EvaluationResult]:
        """Run multiple evaluation tests in parallel.

        Args:
            tests: List of test functions to execute
            model: LLM model to use for testing
            max_workers: Maximum number of parallel workers

        Returns:
            List of EvaluationResult objects
        """
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self.run_test, test, model): test for test in tests
            }

            # Process results as they complete
            for future in as_completed(future_to_test):
                try:
                    result = future.result()
                    results.append(result)

                    # Print progress
                    status = "PASS" if result.success else "FAIL"
                    print(f"{status}: {result.test_name} ({result.duration_ms}ms)")
                    if not result.success:
                        print(f"  Error: {result.error_message}")
                except Exception as e:
                    test_func = future_to_test[future]
                    test_name = getattr(test_func, "__name__", str(test_func))
                    print(f"FAIL: {test_name} (execution error: {e})")
                    results.append(
                        EvaluationResult(
                            test_name=test_name,
                            success=False,
                            duration_ms=0,
                            error_message=f"Execution error: {str(e)}",
                        )
                    )

        return results

    def print_summary(self, results: list[EvaluationResult]):
        """Print a summary of evaluation results.

        Args:
            results: List of EvaluationResult objects
        """
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        success_rate = (passed / len(results)) * 100 if results else 0.0

        print("\n=== Benchmark Summary ===")
        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {success_rate:.1f}%")

        if failed > 0:
            print("\nFailed tests:")
            for result in results:
                if not result.success:
                    print(f"  - {result.test_name}: {result.error_message}")

    def main(self):
        """Create Click CLI for running evaluations."""

        @click.command()
        @click.option("--filter", help="Filter tests by name pattern")
        @click.option("--model", default="gemini/gemini-2.5-flash", help="LLM model to use")
        @click.option("--parallel", is_flag=True, help="Run tests in parallel")
        @click.option("--verbose", is_flag=True, help="Enable verbose output")
        def run(filter, model, parallel, verbose):
            """Run evaluation tests."""
            import sys

            # Filter tests if requested
            tests_to_run = self.test_cases
            if filter:
                tests_to_run = [t for t in tests_to_run if filter in t.__name__]
                if not tests_to_run:
                    print(f"No tests found matching filter: {filter}")
                    sys.exit(1)

            if not tests_to_run:
                print("No tests found")
                sys.exit(1)

            print(f"Running {len(tests_to_run)} tests with model: {model}")

            # Create function library if not already set
            if not self.function_library:
                # Build library from registry - assumes tools are already registered
                from tidyllm.registry import REGISTRY
                self.function_library = REGISTRY

            # Run tests
            if parallel:
                results = self.run_tests_parallel(tests_to_run, model)
            else:
                results = self.run_tests(tests_to_run, model)

            # Print summary
            self.print_summary(results)

            # Exit with error code if tests failed
            failed_count = sum(1 for r in results if not r.success)
            sys.exit(1 if failed_count > 0 else 0)

        return run(standalone_mode=False)


def run_evaluations(
    function_library: Registry,
    model: str,
    test_modules: list[Any] | None = None,
) -> list[EvaluationResult]:
    """Convenience function to run evaluations.

    Args:
        function_library: Registry with registered tools
        model: LLM model to use for testing
        test_modules: List of modules containing evaluation tests
        test_path: Path to discover tests from

    Returns:
        List of EvaluationResult objects
    """
    runner = EvaluationRunner(function_library)

    # Discover tests
    if test_modules:
        tests = runner.discover_tests(test_modules)
    else:
        tests = []

    if not tests:
        print("No evaluation tests found")
        return []

    # Run tests
    results = runner.run_tests(tests, model)
    runner.print_summary(results)

    return results


def create_evaluation_cli(module):
    """Create a Click CLI for running evaluations in a module.

    This should be used in evaluation files like:

    if __name__ == "__main__":
        from tidyllm.evaluation import create_evaluation_cli
        create_evaluation_cli(__import__(__name__))
    """
    test_cases = find_test_cases(module)
    runner = EvaluationRunner(test_cases=test_cases)
    return runner.main()

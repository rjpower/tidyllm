#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///

import subprocess
import sys


def run_command(cmd):
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    """Run precommit checks."""
    # Run ruff check with fixes
    ruff_result = run_command(["uv", "run", "ruff", "check", "--fix", "--unsafe-fixes"])
    if ruff_result != 0:
        print("Ruff check failed")
        sys.exit(ruff_result)
    
    # Run pytest
    pytest_result = run_command(["uv", "run", "pytest"])
    if pytest_result != 0:
        print("Pytest failed")
        sys.exit(pytest_result)
    
    print("All checks passed!")

if __name__ == "__main__":
    main()
"""Tests for playwright tools."""

import pytest
from PIL import Image

from tidyllm.tools.playwright import playwright_fetch_html, playwright_fetch_screenshot


def test_fetch_html():
    """Test fetching HTML content."""
    result = playwright_fetch_html("https://example.com")
    
    assert isinstance(result.data, bytes)
    assert len(result.data) > 0
    html_content = result.data.decode()
    assert "<h1>Example Domain</h1>" in html_content


def test_fetch_screenshot():
    """Test taking a screenshot."""
    results = playwright_fetch_screenshot("https://example.com", full_page=False)
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    result = results[0]
    assert isinstance(result.data, bytes)
    assert len(result.data) > 0
    assert result.data[:8] == b'\x89PNG\r\n\x1a\n'  # PNG header


def test_fetch_screenshot_full_page():
    """Test taking a full page screenshot."""
    results = playwright_fetch_screenshot("https://example.com", full_page=True)
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    result = results[0]
    assert isinstance(result.data, bytes)
    assert len(result.data) > 0
    assert result.data[:8] == b'\x89PNG\r\n\x1a\n'  # PNG header


def test_fetch_html_allrecipes():
    """Test fetching HTML from AllRecipes to debug timeout issue."""
    import asyncio
    print(f"Running event loop check...")
    try:
        loop = asyncio.get_running_loop()
        print(f"Found running event loop: {loop}")
    except RuntimeError:
        print("No running event loop found")
    
    try:
        print("Starting fetch...")
        result = playwright_fetch_html("https://www.allrecipes.com/recipe/14415/cobb-salad/")
        assert isinstance(result.data, bytes)
        assert len(result.data) > 0
        html_content = result.data.decode()
        print(f"SUCCESS! HTML length: {len(html_content)}")
        print(f"First 500 chars: {html_content[:500]}")
    except Exception as e:
        print(f"Error fetching AllRecipes: {e}")
        print(f"Error type: {type(e)}")
        raise
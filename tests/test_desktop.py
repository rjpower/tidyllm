"""Tests for desktop automation tools."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext
from tidyllm.tools.desktop import (
    window_focus,
    window_list,
    window_close,
    element_click,
    element_fill,
    element_get_text,
    send_keys,
    mouse_click_coords,
    take_screenshot,
    wait_for_element,
    wait_for_seconds,
    open_file,
    WindowListResult,
    ElementInfo,
)


@pytest.fixture
def tool_context():
    """Create tool context for testing."""
    return ToolContext()


@pytest.fixture
def mock_desktop():
    """Create mock RPA Desktop instance."""
    with patch('tidyllm.tools.desktop._get_desktop') as mock_get_desktop:
        mock_desktop_instance = Mock()
        mock_get_desktop.return_value = mock_desktop_instance
        yield mock_desktop_instance


def test_window_focus_success(tool_context, mock_desktop):
    """Test successful window focus."""
    with set_tool_context(tool_context):
        window_focus("Anki")  # Should not raise exception
        mock_desktop.open_dialog.assert_called_once_with("Anki")


def test_window_focus_failure(tool_context, mock_desktop):
    """Test window focus failure."""
    mock_desktop.open_dialog.side_effect = Exception("Window not found")
    
    with set_tool_context(tool_context):
        with pytest.raises(Exception, match="Window not found"):
            window_focus("NonExistentApp")


def test_window_list_success(tool_context, mock_desktop):
    """Test successful window listing."""
    # Mock window elements
    mock_element1 = Mock()
    mock_element1.name = "Window 1"
    mock_element1.handle = "123"
    mock_element1.process_id = 456
    
    mock_element2 = Mock()
    mock_element2.name = "Window 2"
    mock_element2.handle = "789"
    mock_element2.process_id = 101
    
    mock_desktop.get_window_elements.return_value = [mock_element1, mock_element2]
    
    with set_tool_context(tool_context):
        result = window_list()
        
        assert isinstance(result, WindowListResult)
        assert result.count == 2
        assert len(result.windows) == 2
        assert result.windows[0].title == "Window 1"
        assert result.windows[1].title == "Window 2"


def test_window_list_empty(tool_context, mock_desktop):
    """Test window listing when no windows found."""
    mock_desktop.get_window_elements.return_value = []
    
    with set_tool_context(tool_context):
        result = window_list()
        
        assert isinstance(result, WindowListResult)
        assert result.count == 0
        assert len(result.windows) == 0


def test_window_close_success(tool_context, mock_desktop):
    """Test successful window close."""
    with set_tool_context(tool_context):
        window_close("Anki")  # Should not raise exception
        mock_desktop.open_dialog.assert_called_once_with("Anki")
        mock_desktop.send_keys.assert_called_once_with("alt+f4")


def test_element_click_ocr(tool_context, mock_desktop):
    """Test clicking element using OCR."""
    with set_tool_context(tool_context):
        element_click('ocr:"Import"')  # Should not raise exception
        mock_desktop.click.assert_called_once_with('ocr:"Import"')


def test_element_click_with_window(tool_context, mock_desktop):
    """Test clicking element with window focus."""
    with set_tool_context(tool_context):
        element_click('ocr:"Import"', window_query="Anki")  # Should not raise exception
        mock_desktop.open_dialog.assert_called_once_with("Anki")
        mock_desktop.click.assert_called_once_with('ocr:"Import"')


def test_element_click_name(tool_context, mock_desktop):
    """Test clicking element using name locator."""
    with set_tool_context(tool_context):
        element_click('name:"OK Button"')  # Should not raise exception
        mock_desktop.click.assert_called_once_with('name:OK Button')


def test_element_click_default_ocr(tool_context, mock_desktop):
    """Test clicking element with default OCR."""
    with set_tool_context(tool_context):
        element_click('Submit')  # Should not raise exception
        mock_desktop.click.assert_called_once_with('ocr:"Submit"')


def test_element_fill_success(tool_context, mock_desktop):
    """Test filling element with text."""
    with set_tool_context(tool_context):
        element_fill('ocr:"Username"', 'testuser')  # Should not raise exception
        mock_desktop.click.assert_called_once_with('ocr:"Username"')
        mock_desktop.send_keys.assert_any_call("ctrl+a")
        mock_desktop.send_keys.assert_any_call("testuser")


def test_element_get_text_success(tool_context, mock_desktop):
    """Test getting text from element."""
    mock_element = Mock()
    mock_desktop.get_element.return_value = mock_element
    mock_desktop.get_element_rich_text.return_value = "Sample text"
    mock_desktop.is_element_visible.return_value = True
    mock_desktop.is_element_enabled.return_value = True
    
    with set_tool_context(tool_context):
        result = element_get_text('name:"Status Label"')
        
        assert isinstance(result, ElementInfo)
        assert result.text == "Sample text"
        assert result.visible is True
        assert result.enabled is True


def test_element_get_text_ocr_fallback(tool_context, mock_desktop):
    """Test getting text from OCR element (fallback)."""
    with set_tool_context(tool_context):
        result = element_get_text('ocr:"Status Text"')
        
        assert isinstance(result, ElementInfo)
        assert result.text == 'ocr:"Status Text"'
        assert result.visible is True
        assert result.enabled is True


def test_send_keys_success(tool_context, mock_desktop):
    """Test sending keys successfully."""
    with set_tool_context(tool_context):
        send_keys('ctrl+c')  # Should not raise exception
        mock_desktop.send_keys.assert_called_once_with('ctrl+c')


def test_send_keys_with_window(tool_context, mock_desktop):
    """Test sending keys with window focus."""
    with set_tool_context(tool_context):
        send_keys('Hello', window_query="Notepad")  # Should not raise exception
        mock_desktop.open_dialog.assert_called_once_with("Notepad")
        mock_desktop.send_keys.assert_called_once_with('Hello')


def test_mouse_click_coords_success(tool_context, mock_desktop):
    """Test mouse click at coordinates."""
    with set_tool_context(tool_context):
        mouse_click_coords(100, 200)  # Should not raise exception
        mock_desktop.mouse_click.assert_called_once_with(100, 200)


def test_take_screenshot_success(tool_context, mock_desktop):
    """Test taking screenshot successfully."""
    with set_tool_context(tool_context):
        result = take_screenshot("test.png")
        
        assert result == "test.png"
        mock_desktop.take_screenshot.assert_called_once_with("test.png")


def test_take_screenshot_auto_filename(tool_context, mock_desktop):
    """Test taking screenshot with auto-generated filename."""
    with set_tool_context(tool_context):
        result = take_screenshot()
        
        assert result.startswith("screenshot_")
        assert result.endswith(".png")


def test_wait_for_element_success(tool_context, mock_desktop):
    """Test waiting for element successfully."""
    with set_tool_context(tool_context):
        wait_for_element('ocr:"Import"', timeout=1)  # Should not raise exception
        mock_desktop.click.assert_called_with('ocr:"Import"', dry_run=True)


def test_wait_for_element_timeout(tool_context, mock_desktop):
    """Test waiting for element timeout."""
    mock_desktop.click.side_effect = Exception("Element not found")
    
    with set_tool_context(tool_context):
        with pytest.raises(TimeoutError, match="Timeout waiting for element"):
            wait_for_element('ocr:"NonExistent"', timeout=1)


def test_wait_for_seconds_success(tool_context):
    """Test waiting for seconds."""
    with set_tool_context(tool_context):
        import time
        start = time.time()
        wait_for_seconds(1)  # Should not raise exception
        end = time.time()
        
        assert end - start >= 1


@patch('subprocess.run')
@patch('platform.system')
def test_open_file_macos(mock_system, mock_run, tool_context):
    """Test opening file on macOS."""
    mock_system.return_value = "Darwin"
    
    with set_tool_context(tool_context):
        with patch('pathlib.Path.exists', return_value=True):
            open_file('/path/to/test.apkg')  # Should not raise exception
            mock_run.assert_called_once_with(["open", "/path/to/test.apkg"], check=True)


@patch('subprocess.run')
@patch('platform.system')
def test_open_file_windows(mock_system, mock_run, tool_context):
    """Test opening file on Windows."""
    mock_system.return_value = "Windows"
    
    with set_tool_context(tool_context):
        with patch('pathlib.Path.exists', return_value=True):
            open_file('C:\\path\\to\\test.apkg')  # Should not raise exception
            mock_run.assert_called_once_with(["start", "C:\\path\\to\\test.apkg"], shell=True, check=True)


@patch('subprocess.run')
@patch('platform.system')
def test_open_file_linux(mock_system, mock_run, tool_context):
    """Test opening file on Linux."""
    mock_system.return_value = "Linux"
    
    with set_tool_context(tool_context):
        with patch('pathlib.Path.exists', return_value=True):
            open_file('/path/to/test.apkg')  # Should not raise exception
            mock_run.assert_called_once_with(["xdg-open", "/path/to/test.apkg"], check=True)


def test_open_file_not_found(tool_context):
    """Test opening non-existent file."""
    with set_tool_context(tool_context):
        with pytest.raises(FileNotFoundError, match="File not found"):
            open_file('/nonexistent/file.apkg')


def test_anki_import_workflow(tool_context, mock_desktop):
    """Test the complete Anki import workflow."""
    with set_tool_context(tool_context):
        # Step 1: Open the .apkg file
        with patch('pathlib.Path.exists', return_value=True):
            with patch('subprocess.run'):
                with patch('platform.system', return_value="Darwin"):
                    open_file('/path/to/foo.apkg')  # Should not raise exception
        
        # Step 2: Wait for Anki window to appear
        wait_for_element('ocr:"Import File"', timeout=10)  # Should not raise exception
        
        # Step 3: Focus Anki window
        window_focus("Anki")  # Should not raise exception
        
        # Step 4: Click Import button
        element_click('ocr:"Import"')  # Should not raise exception
        
        # Step 5: Wait for import to finish
        wait_for_element('ocr:"Import complete"', timeout=60)  # Should not raise exception
        
        # Step 6: Close Anki
        window_close("Anki")  # Should not raise exception
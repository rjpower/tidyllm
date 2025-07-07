"""Tests for desktop automation tools."""

from unittest.mock import Mock, patch

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext
from tidyllm.tools.desktop import (
    ElementInfo,
    WindowListResult,
    element_click,
    element_fill,
    element_get_text,
    mouse_click_coords,
    open_file,
    send_keys,
    take_screenshot,
    wait_for_element,
    wait_for_seconds,
    window_close,
    window_focus,
    window_list,
)


@pytest.fixture
def tool_context():
    """Create tool context for testing."""
    return ToolContext()


@pytest.fixture
def mock_applescript():
    """Create mock AppleScript execution."""
    with patch('tidyllm.tools.desktop.subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        yield mock_run


def test_window_focus_success(tool_context, mock_applescript):
    """Test successful window focus."""
    with set_tool_context(tool_context):
        window_focus("Anki")  # Should not raise exception
        mock_applescript.assert_called_with(['osascript', '-e', 'tell application "Anki" to activate'], capture_output=True)


def test_window_focus_failure(tool_context, mock_applescript):
    """Test window focus failure."""
    mock_applescript.return_value = Mock(returncode=1)
    
    with set_tool_context(tool_context):
        with pytest.raises(ValueError, match="Could not focus window/application"):
            window_focus("NonExistentApp")


def test_window_list_success(tool_context, mock_applescript):
    """Test successful window listing."""
    # Mock AppleScript output
    mock_applescript.return_value = Mock(returncode=0, stdout="App1|Window 1, App2|Window 2")
    
    with set_tool_context(tool_context):
        result = window_list()
        
        assert isinstance(result, WindowListResult)
        assert result.count == 2
        assert len(result.windows) == 2
        assert result.windows[0].title == "App1 - Window 1"
        assert result.windows[1].title == "App2 - Window 2"


def test_window_list_empty(tool_context, mock_applescript):
    """Test window listing when no windows found."""
    mock_applescript.return_value = Mock(returncode=0, stdout="")
    
    with set_tool_context(tool_context):
        result = window_list()
        
        assert isinstance(result, WindowListResult)
        assert result.count == 0
        assert len(result.windows) == 0


def test_window_close_success(tool_context, mock_applescript):
    """Test successful window close."""
    with set_tool_context(tool_context):
        window_close("Anki")  # Should not raise exception
        mock_applescript.assert_called_with(['osascript', '-e', 'tell application "Anki" to quit'], capture_output=True)


def test_element_click_success(tool_context, mock_applescript):
    """Test clicking element successfully."""
    with set_tool_context(tool_context):
        element_click('Import')  # Should not raise exception
        # Check that AppleScript was called to click the button
        assert mock_applescript.called


def test_element_click_with_window(tool_context, mock_applescript):
    """Test clicking element with window focus."""
    with set_tool_context(tool_context):
        element_click('Import', window_query="Anki")  # Should not raise exception
        # Check that AppleScript was called for both window focus and click
        assert mock_applescript.call_count >= 2


def test_element_click_failure(tool_context, mock_applescript):
    """Test clicking element failure."""
    mock_applescript.return_value = Mock(returncode=1)
    
    with set_tool_context(tool_context):
        with pytest.raises(ValueError, match="Could not click element"):
            element_click('NonExistentButton')


def test_element_fill_success(tool_context, mock_applescript):
    """Test filling element with text."""
    with set_tool_context(tool_context):
        element_fill('Username', 'testuser')  # Should not raise exception
        assert mock_applescript.called


def test_element_fill_failure(tool_context, mock_applescript):
    """Test filling element failure."""
    mock_applescript.return_value = Mock(returncode=1)
    
    with set_tool_context(tool_context):
        with pytest.raises(ValueError, match="Could not fill text field"):
            element_fill('NonExistentField', 'testuser')


def test_element_get_text_success(tool_context, mock_applescript):
    """Test getting text from element."""
    mock_applescript.return_value = Mock(returncode=0, stdout="Sample text")
    
    with set_tool_context(tool_context):
        result = element_get_text('Status Label')
        
        assert isinstance(result, ElementInfo)
        assert result.text == "Sample text"
        assert result.visible is True
        assert result.enabled is True


def test_element_get_text_failure(tool_context, mock_applescript):
    """Test getting text from element failure."""
    mock_applescript.return_value = Mock(returncode=1)
    
    with set_tool_context(tool_context):
        result = element_get_text('NonExistentLabel')
        
        assert isinstance(result, ElementInfo)
        assert result.text == ""
        assert result.visible is False
        assert result.enabled is False


def test_send_keys_success(tool_context, mock_applescript):
    """Test sending keys successfully."""
    with set_tool_context(tool_context):
        send_keys('cmd+c')  # Should not raise exception
        assert mock_applescript.called


def test_send_keys_with_window(tool_context, mock_applescript):
    """Test sending keys with window focus."""
    with set_tool_context(tool_context):
        send_keys('Hello', window_query="Notepad")  # Should not raise exception
        assert mock_applescript.call_count >= 2


def test_mouse_click_coords_success(tool_context, mock_applescript):
    """Test mouse click at coordinates."""
    with set_tool_context(tool_context):
        mouse_click_coords(100, 200)  # Should not raise exception
        assert mock_applescript.called


def test_take_screenshot_success(tool_context, mock_applescript):
    """Test taking screenshot successfully."""
    with set_tool_context(tool_context):
        result = take_screenshot("test.png")
        
        assert result.endswith("test.png")
        assert mock_applescript.called


def test_take_screenshot_auto_filename(tool_context, mock_applescript):
    """Test taking screenshot with auto-generated filename."""
    with set_tool_context(tool_context):
        result = take_screenshot()
        
        assert "screenshot_" in result
        assert result.endswith(".png")


def test_wait_for_element_success(tool_context, mock_applescript):
    """Test waiting for element successfully."""
    mock_applescript.return_value = Mock(returncode=0, stdout="true")
    
    with set_tool_context(tool_context):
        wait_for_element('Import', timeout=1)  # Should not raise exception
        assert mock_applescript.called


def test_wait_for_element_timeout(tool_context, mock_applescript):
    """Test waiting for element timeout."""
    mock_applescript.return_value = Mock(returncode=1, stdout="false")
    
    with set_tool_context(tool_context):
        with pytest.raises(TimeoutError, match="Timeout waiting for element"):
            wait_for_element('NonExistent', timeout=1)


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


def test_anki_import_workflow(tool_context, mock_applescript):
    """Test the complete Anki import workflow."""
    # Mock successful AppleScript calls
    mock_applescript.return_value = Mock(returncode=0, stdout="true")
    
    with set_tool_context(tool_context):
        # Step 1: Open the .apkg file
        with patch('pathlib.Path.exists', return_value=True):
            with patch('subprocess.run'):
                with patch('platform.system', return_value="Darwin"):
                    open_file('/path/to/foo.apkg')  # Should not raise exception
        
        # Step 2: Wait for Anki window to appear
        wait_for_element('Import File', timeout=10)  # Should not raise exception
        
        # Step 3: Focus Anki window
        window_focus("Anki")  # Should not raise exception
        
        # Step 4: Click Import button
        element_click('Import')  # Should not raise exception
        
        # Step 5: Wait for import to finish
        wait_for_element('Import complete', timeout=60)  # Should not raise exception
        
        # Step 6: Close Anki
        window_close("Anki")  # Should not raise exception
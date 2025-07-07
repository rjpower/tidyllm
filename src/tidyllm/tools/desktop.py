"""Desktop automation tools using AppleScript for macOS."""

import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from tidyllm.registry import register


class WindowInfo(BaseModel):
    """Information about a window."""
    title: str = Field(description="Window title")
    handle: str = Field(description="Window handle/identifier")
    pid: int | None = Field(default=None, description="Process ID")


class WindowListResult(BaseModel):
    """Result of listing windows."""
    windows: list[WindowInfo] = Field(description="List of windows")
    count: int = Field(description="Number of windows found")


class DesktopResult(BaseModel):
    """Generic result for desktop operations."""
    success: bool = Field(description="Whether operation was successful")
    message: str = Field(description="Result message")


class ElementInfo(BaseModel):
    """Information about a found element."""
    text: str = Field(description="Element text content")
    visible: bool = Field(description="Whether element is visible")
    enabled: bool = Field(description="Whether element is enabled")


class ScreenshotResult(BaseModel):
    """Result of taking a screenshot."""
    success: bool = Field(description="Whether screenshot was taken")
    path: str = Field(description="Path to screenshot file")
    message: str = Field(description="Result message")


def _run_applescript(script: str) -> str:
    """Run AppleScript and return the result."""
    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()


def _get_macos_windows():
    """Get window list on macOS using AppleScript."""
    script = '''
    tell application "System Events"
        set windowList to {}
        set appList to every application process whose visible is true
        repeat with appProc in appList
            set appName to name of appProc
            try
                set windowTitles to title of every window of appProc
                repeat with windowTitle in windowTitles
                    if windowTitle is not "" then
                        set end of windowList to (appName & "|" & windowTitle)
                    end if
                end repeat
            on error
                -- Skip apps that don't have windows or access issues
            end try
        end repeat
        return windowList
    end tell
    '''
    
    result = _run_applescript(script)
    
    # Parse the AppleScript output
    windows = []
    if result:
        # Split by comma and process each window
        items = result.split(', ')
        for item in items:
            item = item.strip()
            if '|' in item:
                app_name, window_title = item.split('|', 1)
                windows.append({
                    'title': f"{app_name} - {window_title}",
                    'app_name': app_name,
                    'window_title': window_title,
                    'handle': f"{app_name}:{window_title}",
                    'pid': None
                })
    
    return windows


@register()
def window_focus(query: str):
    """Focus a window by title or application name.
    
    Args:
        query: Window title or application name to match
        
    Example: window_focus("Anki")
    """
    # Try to activate application directly first
    script = f'tell application "{query}" to activate'
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    
    if result.returncode != 0:
        # If direct activation fails, try as a window title
        script = f'''
        tell application "System Events"
            set frontmost of first application process whose name contains "{query}" to true
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True)
        
        if result.returncode != 0:
            raise ValueError(f"Could not focus window/application: {query}")


@register()
def window_list() -> WindowListResult:
    """List all open windows.
    
    Returns:
        WindowListResult with list of windows and count
        
    Example: window_list()
    """
    window_list_data = _get_macos_windows()
    windows = []
    for window_data in window_list_data:
        windows.append(WindowInfo(
            title=window_data.get('title', ''),
            handle=str(window_data.get('handle', '')),
            pid=window_data.get('pid', None)
        ))
    return WindowListResult(windows=windows, count=len(windows))


@register()
def window_close(query: str):
    """Close a window by title or application name.
    
    Args:
        query: Window title or application name to match
        
    Example: window_close("Anki")
    """
    # Try to quit application
    script = f'tell application "{query}" to quit'
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    
    if result.returncode != 0:
        # If quitting fails, try to close window using System Events
        script = f'''
        tell application "System Events"
            tell application process "{query}"
                click button "close" of front window
            end tell
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True)
        
        if result.returncode != 0:
            raise ValueError(f"Could not close window/application: {query}")


@register()
def element_click(query: str, window_query: str | None = None):
    """Click an element by button name or UI element.
    
    Args:
        query: Element selector (button name, UI element name)
        window_query: Optional window to focus first
        
    Example: element_click('OK')
    Example: element_click('Import', 'Anki')
    """
    # Focus window if specified
    if window_query:
        window_focus(window_query)
    
    # Click UI element using AppleScript
    script = f'''
    tell application "System Events"
        click button "{query}" of front window of (first application process whose frontmost is true)
    end tell
    '''
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    
    if result.returncode != 0:
        # Try as a menu item
        script = f'''
        tell application "System Events"
            click menu item "{query}" of front window of (first application process whose frontmost is true)
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True)
        
        if result.returncode != 0:
            raise ValueError(f"Could not click element: {query}")


@register()
def element_fill(query: str, text: str, window_query: str | None = None):
    """Fill text into a text field.
    
    Args:
        query: Text field name or identifier
        text: Text to fill into the element
        window_query: Optional window to focus first
        
    Example: element_fill('Username', 'myusername')
    """
    # Focus window if specified
    if window_query:
        window_focus(window_query)
    
    # Click the text field and type text
    script = f'''
    tell application "System Events"
        click text field "{query}" of front window of (first application process whose frontmost is true)
        keystroke "a" using command down
        keystroke "{text}"
    end tell
    '''
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    
    if result.returncode != 0:
        raise ValueError(f"Could not fill text field: {query}")


@register()
def element_get_text(query: str, window_query: str | None = None) -> ElementInfo:
    """Get text content from a UI element.
    
    Args:
        query: Element name or identifier
        window_query: Optional window to focus first
        
    Returns:
        ElementInfo with text content and properties
        
    Example: element_get_text('Status Label')
    """
    # Focus window if specified
    if window_query:
        window_focus(window_query)
    
    # Get text from UI element
    script = f'''
    tell application "System Events"
        value of text field "{query}" of front window of (first application process whose frontmost is true)
    end tell
    '''
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    
    if result.returncode == 0:
        text = result.stdout.strip()
        return ElementInfo(text=text, visible=True, enabled=True)
    else:
        # Try as static text
        script = f'''
        tell application "System Events"
            value of static text "{query}" of front window of (first application process whose frontmost is true)
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        
        if result.returncode == 0:
            text = result.stdout.strip()
            return ElementInfo(text=text, visible=True, enabled=True)
        else:
            return ElementInfo(text="", visible=False, enabled=False)


@register()
def send_keys(keys: str, window_query: str | None = None):
    """Send keyboard input to the active window or specified window.
    
    Args:
        keys: Keys to send (supports 'cmd+c', 'cmd+v', etc., or regular text)
        window_query: Optional window to focus first
        
    Example: send_keys('cmd+c')
    Example: send_keys('Hello World')
    """
    # Focus window if specified
    if window_query:
        window_focus(window_query)
    
    # Handle key combinations vs regular text
    if '+' in keys and any(modifier in keys.lower() for modifier in ['cmd', 'control', 'option', 'shift']):
        # Parse key combination
        parts = keys.split('+')
        modifier_map = {
            'cmd': 'command',
            'ctrl': 'control',
            'alt': 'option',
            'option': 'option',
            'shift': 'shift'
        }
        
        modifiers = []
        key = parts[-1]
        
        for part in parts[:-1]:
            part = part.strip().lower()
            if part in modifier_map:
                modifiers.append(modifier_map[part])
        
        if modifiers:
            modifier_str = ' down, '.join(modifiers) + ' down'
            script = f'''
            tell application "System Events"
                keystroke "{key}" using {{{modifier_str}}}
            end tell
            '''
        else:
            script = f'''
            tell application "System Events"
                keystroke "{key}"
            end tell
            '''
    else:
        # Regular text
        script = f'''
        tell application "System Events"
            keystroke "{keys}"
        end tell
        '''
    
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"Could not send keys: {keys}")


@register()
def mouse_click_coords(x: int, y: int):
    """Click at specific screen coordinates.
    
    Args:
        x: X coordinate on screen
        y: Y coordinate on screen
        
    Example: mouse_click_coords(100, 200)
    """
    script = f'''
    tell application "System Events"
        click at {{{x}, {y}}}
    end tell
    '''
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"Could not click at coordinates: ({x}, {y})")


@register()
def take_screenshot(filename: str | None = None) -> str:
    """Take a screenshot of the screen.
    
    Args:
        filename: Optional filename for screenshot (defaults to timestamp)
        
    Returns:
        Path to the screenshot file
        
    Example: take_screenshot('anki_import.png')
    """
    if filename is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
    
    # Ensure filename has .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    
    screenshot_path = Path(filename).absolute()
    
    # Use macOS screencapture command
    result = subprocess.run(['screencapture', str(screenshot_path)], capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"Could not take screenshot: {filename}")
    
    return str(screenshot_path)


@register()
def wait_for_element(query: str, timeout: int = 30, window_query: str | None = None):
    """Wait for an element to appear on screen.
    
    Args:
        query: Element name or button name
        timeout: Maximum time to wait in seconds
        window_query: Optional window to focus first
        
    Example: wait_for_element('Import complete', 60)
    """
    # Focus window if specified
    if window_query:
        window_focus(window_query)
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Try to find the element
        script = f'''
        tell application "System Events"
            exists button "{query}" of front window of (first application process whose frontmost is true)
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip() == "true":
            return  # Element found
        
        # Also try as text field
        script = f'''
        tell application "System Events"
            exists text field "{query}" of front window of (first application process whose frontmost is true)
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip() == "true":
            return  # Element found
        
        # Wait a bit before trying again
        time.sleep(1)
    
    # Timeout reached
    raise TimeoutError(f"Timeout waiting for element: {query}")


@register()
def wait_for_seconds(seconds: int):
    """Wait for a specified number of seconds.
    
    Args:
        seconds: Number of seconds to wait
        
    Example: wait_for_seconds(5)
    """
    time.sleep(seconds)


@register()
def open_file(file_path: str):
    """Open a file using the system's default application.
    
    Args:
        file_path: Path to the file to open
        
    Example: open_file('/path/to/foo.apkg')
    """
    import platform
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use system-specific open command
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", str(path)], check=True)
    elif platform.system() == "Windows":
        subprocess.run(["start", str(path)], shell=True, check=True)
    else:  # Linux
        subprocess.run(["xdg-open", str(path)], check=True)
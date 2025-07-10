"""Web scraping tools using Playwright for fetching HTML and screenshots."""

import asyncio
import base64
from collections.abc import Awaitable, Callable
from typing import TypeVar

from playwright.async_api import Page, async_playwright

from tidyllm.registry import register
from tidyllm.types.part import HtmlPart, PngPart

T = TypeVar("T")


async def _fetch(url: str, callback: Callable[[Page], Awaitable[T]]) -> T:
    """Generic fetch helper that handles browser lifecycle.

    Args:
        url: URL to fetch
        callback: Async function that takes a Page and returns result

    Returns:
        Result from callback
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Use domcontentloaded as default - networkidle is problematic with ad-heavy sites
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)

        result = await callback(page)

        await browser.close()

    return result


async def _fetch_html_async(url: str) -> HtmlPart:
    """Fetch HTML content from a URL."""

    async def extract_content(page: Page) -> HtmlPart:
        content = await page.content()
        return HtmlPart(data=base64.b64encode(content.encode()))

    return await _fetch(url, extract_content)


async def _fetch_screenshot_async(url: str, full_page: bool) -> list[PngPart]:
    async def take_screenshot(page: Page) -> list[PngPart]:
        # Set viewport width to 1024
        await page.set_viewport_size({"width": 1024, "height": 768})
        
        screenshot_bytes = await page.screenshot(full_page=full_page, type="png")
        
        # Check image height and slice if needed
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(screenshot_bytes))
        width, height = image.size
        
        if height <= 1024:
            return [PngPart.from_bytes(screenshot_bytes)]
        
        # Slice image into 1024px height chunks
        parts = []
        for y in range(0, height, 1024):
            box = (0, y, width, min(y + 1024, height))
            slice_img = image.crop(box)
            
            # Convert slice back to bytes
            slice_buffer = io.BytesIO()
            slice_img.save(slice_buffer, format='PNG')
            slice_bytes = slice_buffer.getvalue()
            
            parts.append(PngPart.from_bytes(slice_bytes))
        
        return parts

    return await _fetch(url, take_screenshot)


def _run_async(coro):
    """Run async coroutine, handling existing event loop."""
    try:
        asyncio.get_running_loop()
        # We're in an async context but want to block - use new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop running, so we can use asyncio.run directly
        return asyncio.run(coro)


@register()
def playwright_fetch_screenshot(url: str, full_page: bool = False) -> list[PngPart]:
    """Take a screenshot of a web page.

    Args:
        url: The URL to capture
        full_page: Whether to capture the full scrollable page or only the viewable region.
    """
    return _run_async(_fetch_screenshot_async(url, full_page))


@register()
def playwright_fetch_html(url: str) -> HtmlPart:
    """Fetch the html content of `url` using Playwright."""
    return _run_async(_fetch_html_async(url))


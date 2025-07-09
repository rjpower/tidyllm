#!/usr/bin/env python3
"""PDF to Note transcription app for handwritten notes.

This app extracts pages from PDF files as images and transcribes handwritten notes
using vision models, generating well-formatted markdown notes with frontmatter.
"""

import base64
import io
import tempfile
from pathlib import Path

import pypdfium2 as pdfium
from pydantic import BaseModel, Field
from rich.console import Console

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.llm import completion_with_schema
from tidyllm.registry import register
from tidyllm.source import SourceLike, read_bytes
from tidyllm.tools.context import ToolContext
from tidyllm.tools.notes import NoteAddArgs, note_add

console = Console()


class ImageData(BaseModel):
    """Image data extracted from PDF."""

    page_number: int
    content_base64: str = Field(description="Base64 encoded image content")
    mime_type: str = "image/jpeg"
    width: int
    height: int

    @property
    def content(self) -> bytes:
        """Get binary content from base64."""
        return base64.b64decode(self.content_base64)


class TranscriptionResponse(BaseModel):
    """Structured response from LLM transcription."""

    title: str = Field(description="Title for the note")
    tags: list[str] = Field(description="Tags for the note")
    content: str = Field(description="Markdown content with frontmatter")


@register()
def extract_pdf_images(
    pdf_source: SourceLike,
    image_width: int = 768,
    image_height: int = 1084,
    jpeg_quality: int = 85,
) -> list[ImageData]:
    """Extract pages from PDF as images.

    Args:
        pdf_source: PDF source (file path, bytes, URL, etc.)

    Returns:
        List of ImageData objects
    """
    console.print("[yellow]Extracting images from PDF source[/yellow]")

    # Read PDF bytes from source
    pdf_bytes = read_bytes(pdf_source)

    pdf = pdfium.PdfDocument(pdf_bytes)
    images = []

    for page_num in range(len(pdf)):
        page = pdf.get_page(page_num)

        # Render page as bitmap
        bitmap = page.render(scale=1.0, rotation=0, crop=(0, 0, 0, 0))

        # Convert to PIL Image and resize
        pil_image = bitmap.to_pil()
        pil_image = pil_image.resize((image_width, image_height))

        # Convert to JPEG bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG", quality=jpeg_quality)
        img_bytes.seek(0)

        images.append(
            ImageData(
                page_number=page_num + 1,
                content_base64=base64.b64encode(img_bytes.getvalue()).decode(),
                width=image_width,
                height=image_height,
            )
        )

        console.print(f"[green]Extracted page {page_num + 1}[/green]")

    console.print(f"[green]Extracted {len(images)} pages total[/green]")
    return images


TRANSCRIPTION_PROMPT = """
You are transcribing handwritten notes from pages of a PDF document.

Please transcribe the content into well-formatted markdown with the following requirements:

1. **Structure**: Create a comprehensive markdown document that combines all pages
2. **Frontmatter**: Include YAML frontmatter with title, tags, and date
3. **Formatting**: Use proper markdown syntax including:
   - Headers (# ## ###) for sections
   - **Bold** and *italic* text where appropriate
   - Lists (bullet points and numbered lists)
   - Code blocks for equations or formulas
   - Tables if applicable

4. **Enhancement**: Where appropriate, add:
   - Wikipedia links for significant concepts/terms using [term](https://en.wikipedia.org/wiki/Term) format
   - Tufte-style margin notes using footnotes[^1] for clarifications or additional context
   - Mathematical notation using LaTeX format: $inline$ or $$block$$

5. **Academic Style**: Follow academic writing conventions:
   - Clear section headers
   - Proper citations if references are mentioned
   - Logical flow and organization

Example output structure:
```markdown
---
title: "Your Generated Title"
tags: ["tag1", "tag2", "tag3"]
date: 2025-07-08
---

# Main Title

## Section 1

Content here with proper formatting...

[Important concept](https://en.wikipedia.org/wiki/Important_concept) is discussed here.

- Bullet point 1
- Bullet point 2

Mathematical formula: $E = mc^2$

Some text with a margin note about clarification.[^1]

[^1]: This is a margin note providing additional context or clarification.
```

Transcribe ALL text visible in the images, maintaining the logical flow and structure of the original notes.
"""


def transcribe_images_to_markdown(images: list[ImageData]) -> TranscriptionResponse:
    """Transcribe images using vision model to generate markdown.

    Args:
        images: List of image data from PDF

    Returns:
        TranscriptionResponse with title, tags, and content
    """
    ctx = get_tool_context()

    console.print(
        f"[yellow]Transcribing {len(images)} images using vision model...[/yellow]"
    )

    # Create message with images
    image_contents = []
    for img in images:
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.mime_type};base64,{img.content_base64}"
                },
            }
        )

    # Prepare messages for LLM
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIPTION_PROMPT},
                *image_contents,
            ],
        }
    ]

    # Call LLM with structured response
    response = completion_with_schema(
        model=ctx.config.fast_model,
        messages=messages,
        response_schema=TranscriptionResponse,
    )

    console.print("[green]Transcription completed[/green]")
    console.print(f"[blue]Title:[/blue] {response.title}")
    console.print(f"[blue]Tags:[/blue] {', '.join(response.tags)}")

    return response


@register()
def pdf_to_markdown(
    pdf_source: SourceLike,
    image_width: int = 768,
    image_height: int = 1084,
    jpeg_quality: int = 85,
):
    console.print("[bold blue]Converting PDF to note from source[/bold blue]")
    images = extract_pdf_images(pdf_source, image_width, image_height, jpeg_quality)
    return transcribe_images_to_markdown(images)


@register()
def pdf_to_note(
    pdf_source: SourceLike,
    title: str | None = None,
    tags: tuple[str] = tuple(),
    image_width: int = 768,
    image_height: int = 1084,
    jpeg_quality: int = 85,
) -> str:
    transcription = pdf_to_markdown(pdf_source, image_width, image_height, jpeg_quality)
    final_title = title if title else transcription.title
    final_tags = tags if tags else transcription.tags
    note_path = note_add(
        NoteAddArgs(content=transcription.content, title=final_title, tags=final_tags)
    )
    console.print("[bold green]Pipeline completed![/bold green]")
    return note_path


if __name__ == "__main__":
    functions = [
        extract_pdf_images,
        pdf_to_markdown,
        pdf_to_note,
    ]
    cli_main(functions, context_cls=ToolContext)

#!/usr/bin/env python3
"""PDF to Note transcription app for handwritten notes.

This app extracts pages from PDF files as images and transcribes handwritten notes
using vision models, generating well-formatted markdown notes with frontmatter.
"""

from pydantic import BaseModel, Field
from rich.console import Console

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.llm import completion_with_schema
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext
from tidyllm.tools.notes import NoteAddArgs, note_add
from tidyllm.types.part import Part, is_pdf_part
from tidyllm.types.part.image import ImagePart

console = Console()


class TranscriptionResponse(BaseModel):
    """Structured response from LLM transcription."""

    title: str = Field(description="Title for the note")
    tags: list[str] = Field(description="Tags for the note")
    content: str = Field(description="Markdown content with frontmatter")


@register()
def extract_pdf_images(
    pdf_url: str,
    image_width: int = 768,
    image_height: int = 1084,
    jpeg_quality: int = 85,
) -> list[ImagePart]:
    """Extract pages from PDF as images.

    Args:
        pdf_url: PDF URL (file://, gdrive://, https://, etc.)

    Returns:
        List of ImagePart objects
    """
    console.print("[yellow]Extracting images from PDF source[/yellow]")

    # Get PDF Part from URL
    pdf_parts = Part.from_url(pdf_url)
    pdf_part = next(iter(pdf_parts))  # Get first (and likely only) part
    
    # Ensure we have a PdfPart
    if not is_pdf_part(pdf_part):
        raise ValueError(f"Expected PDF part, got {type(pdf_part)}")
    
    # Extract images using PdfPart
    images = pdf_part.extract_images(image_width, image_height, jpeg_quality)
    
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


def transcribe_images_to_markdown(images: list[ImagePart]) -> TranscriptionResponse:
    """Transcribe images using vision model to generate markdown.

    Args:
        images: List of ImagePart objects from PDF

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
        # Convert ImagePart to base64 data URL
        image_base64 = img.to_base64()
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.mime_type};base64,{image_base64}"
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
    pdf_url: str,
    image_width: int = 768,
    image_height: int = 1084,
    jpeg_quality: int = 85,
):
    console.print("[bold blue]Converting PDF to note from URL[/bold blue]")
    images = extract_pdf_images(pdf_url, image_width, image_height, jpeg_quality)
    return transcribe_images_to_markdown(images)


@register()
def pdf_to_note(
    pdf_url: str,
    title: str | None = None,
    tags: tuple[str] = tuple(),
    image_width: int = 768,
    image_height: int = 1084,
    jpeg_quality: int = 85,
) -> str:
    transcription = pdf_to_markdown(pdf_url, image_width, image_height, jpeg_quality)
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

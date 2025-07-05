"""User notes management tools."""

import re
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class Note(BaseModel):
    """A note with metadata."""
    file_path: Path
    title: str
    tags: list[str] = Field(default_factory=list)
    content_preview: str
    created_at: datetime
    updated_at: datetime


# Add Note Tool
class NoteAddArgs(BaseModel):
    """Arguments for adding a note."""
    content: str = Field(description="Note content (markdown)")
    title: str | None = Field(None, description="Note title")
    tags: list[str] = Field(default_factory=list, description="Tags for the note")


def parse_frontmatter(content: str) -> tuple[dict[str, str | list[str]], str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---\n"):
        return {}, content
        
    try:
        end_match = re.search(r'\n---\n', content[4:])
        if not end_match:
            return {}, content
            
        frontmatter_text = content[4:end_match.start() + 4]
        body = content[end_match.end() + 4:]
        
        # Simple YAML parsing for basic key-value pairs
        frontmatter = {}
        for line in frontmatter_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle list format [tag1, tag2]
                if value.startswith('[') and value.endswith(']'):
                    items = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
                    frontmatter[key] = items
                else:
                    frontmatter[key] = value.strip('"\'')
                    
        return frontmatter, body
    except Exception:
        return {}, content


def create_frontmatter(title: str, tags: list[str]) -> str:
    """Create YAML frontmatter."""
    tags_str = ', '.join(f'"{tag}"' for tag in tags)
    return f"""---
title: "{title}"
tags: [{tags_str}]
created: {datetime.now().isoformat()}
---

"""


def sanitize_filename(title: str) -> str:
    """Create a safe filename from title."""
    # Remove invalid characters
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with underscores
    safe_title = safe_title.replace(' ', '_')
    # Limit length
    if len(safe_title) > 50:
        safe_title = safe_title[:50]
    # Add timestamp if empty or too short
    if len(safe_title) < 3:
        safe_title = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return safe_title


def _parse_note_file(file_path: Path) -> Note | None:
    """Parse a note file and extract metadata."""
    try:
        content = file_path.read_text()
        frontmatter, body = parse_frontmatter(content)

        # Extract title from frontmatter or use filename
        title = frontmatter.get("title", file_path.stem.replace("_", " "))

        # Extract tags from frontmatter
        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        # Get file timestamps
        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
        updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Create content preview
        content_preview = body[:200].replace("\n", " ").strip()

        return Note(
            file_path=str(file_path),
            title=title,
            tags=tags,
            content_preview=content_preview,
            created_at=created_at,
            updated_at=updated_at,
        )
    except Exception:
        return None


@register()
def note_add(args: NoteAddArgs) -> str:
    """Add a new note with markdown and frontmatter.

    Example usage: note_add({"content": "This is my note content", "title": "My Important Note", "tags": ["personal", "ideas"]})
    """
    ctx = get_tool_context()
    notes_dir = ctx.config.ensure_notes_dir()

    title = args.title or f"Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    tags = args.tags

    # Create filename
    filename = sanitize_filename(title) + ".md"
    file_path = notes_dir / filename

    # Ensure unique filename
    counter = 1
    while file_path.exists():
        filename = f"{sanitize_filename(title)}_{counter}.md"
        file_path = notes_dir / filename
        counter += 1

    # Write file with frontmatter
    full_content = create_frontmatter(title, tags) + args.content
    file_path.write_text(full_content)

    return str(file_path)


class NoteSearchResult(BaseModel):
    """Result of note search."""
    notes: list[Note] = Field(default_factory=list)
    count: int = 0


@register()
def note_search(query: str) -> NoteSearchResult:
    """Search notes by content and filename using ripgrep and find.

    Args:
        query: Search query
        
    Example usage: note_search("important meeting")
    """
    ctx = get_tool_context()
    notes_dir = ctx.config.ensure_notes_dir()

    found_files = set()

    # Search content using ripgrep
    rg_result = subprocess.run(
        ["rg", "-l", "-i", "--type", "md", query, str(notes_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    if rg_result.returncode == 0:  # Found content matches
        file_paths = rg_result.stdout.strip().split("\n")
        for file_path_str in file_paths:
            if file_path_str:
                found_files.add(file_path_str)

    # Search filenames using find
    find_result = subprocess.run(
        [
            "find",
            str(notes_dir),
            "-iname",
            f"*{query}*",
            "-type",
            "f",
            "-name",
            "*.md",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if find_result.returncode == 0:  # Found filename matches
        file_paths = find_result.stdout.strip().split("\n")
        for file_path_str in file_paths:
            if file_path_str:
                found_files.add(file_path_str)

    # Parse all found files
    notes_list = []
    for file_path_str in found_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            note = _parse_note_file(file_path)
            if note:
                notes_list.append(note)

    return NoteSearchResult(notes=notes_list, count=len(notes_list))


class NoteListResult(BaseModel):
    """Result of listing notes."""
    notes: list[Note] = Field(default_factory=list)
    count: int = 0


@register()
def note_list(tags: list[str] | None = None, limit: int = 50) -> NoteListResult:
    """List all notes, optionally filtered by tags.

    Args:
        tags: Filter by tags (optional)
        limit: Maximum notes to return (default: 50)
        
    Example usage: note_list(["work"], 20) or note_list() for all notes
    """
    ctx = get_tool_context()
    notes_dir = ctx.config.ensure_notes_dir()

    # Find all markdown files
    md_files = list(notes_dir.glob("*.md"))
    md_files.sort(
        key=lambda f: f.stat().st_mtime, reverse=True
    )  # Sort by modification time

    notes_list = []
    for file_path in md_files[:limit]:
        note = _parse_note_file(file_path)
        if note:
            # Filter by tags if specified
            if tags:
                if any(tag in note.tags for tag in tags):
                    notes_list.append(note)
            else:
                notes_list.append(note)

    return NoteListResult(notes=notes_list, count=len(notes_list))


@register()
def note_open(query: str) -> str:
    """Open and display a note by title or filename.

    Args:
        query: Note title or filename to open
        
    Example usage: note_open("my important note")
    """
    ctx = get_tool_context()
    notes_dir = ctx.config.ensure_notes_dir()

    # First try exact filename match
    filename = query if query.endswith(".md") else f"{query}.md"
    file_path = notes_dir / filename

    if file_path.exists():
        content = file_path.read_text()
        return content

    # Try fuzzy search by title
    md_files = list(notes_dir.glob("*.md"))
    for file_path in md_files:
        note = _parse_note_file(file_path)
        if note and query.lower() in note.title.lower():
            content = file_path.read_text()
            return content

    raise FileNotFoundError(f"Note not found: {query}")


class NoteRecentResult(BaseModel):
    """Result of listing recent notes."""

    notes: list[Note] = Field(default_factory=list)
    count: int = 0


@register()
def note_recent(limit: int = 10) -> NoteRecentResult:
    """List recently modified notes.

    Args:
        limit: Number of recent notes to show (default: 10)
        
    Example usage: note_recent(5)
    """
    ctx = get_tool_context()
    notes_dir = ctx.config.ensure_notes_dir()

    # Find all markdown files and sort by modification time
    md_files = list(notes_dir.glob("*.md"))
    md_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    notes_list = []
    for file_path in md_files[:limit]:
        note = _parse_note_file(file_path)
        if note:
            notes_list.append(note)

    return NoteRecentResult(notes=notes_list, count=len(notes_list))


class NoteTagsResult(BaseModel):
    """Result of listing tags."""

    tags: list[str] = Field(default_factory=list)
    count: int = 0


@register()
def note_tags() -> NoteTagsResult:
    """List all unique tags used in notes.

    Example usage: note_tags()
    """
    ctx = get_tool_context()
    notes_dir = ctx.config.ensure_notes_dir()

    all_tags = set()
    md_files = list(notes_dir.glob("*.md"))

    for file_path in md_files:
        note = _parse_note_file(file_path)
        if note:
            all_tags.update(note.tags)

    sorted_tags = sorted(list(all_tags))

    return NoteTagsResult(tags=sorted_tags, count=len(sorted_tags))


if __name__ == "__main__":
    cli_main(
        [note_add, note_search, note_list, note_open, note_recent, note_tags],
        default_function="note_recent",
        context_cls=ToolContext,
    )

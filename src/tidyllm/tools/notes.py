"""User notes management tools."""

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from tidyllm.multi_cli import simple_cli_main
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext
from tidyllm.tools.db import init_database, json_decode, json_encode, row_to_dict
from tidyllm.context import get_tool_context


class Note(BaseModel):
    """A note with metadata."""
    file_path: str  # Path as string for JSON serialization
    title: str
    tags: list[str] = Field(default_factory=list)
    content_preview: str
    created_at: str  # ISO format string
    updated_at: str  # ISO format string


# Add Note Tool
class NoteAddArgs(BaseModel):
    """Arguments for adding a note."""
    content: str = Field(description="Note content (markdown)")
    title: str | None = Field(None, description="Note title")
    tags: list[str] = Field(default_factory=list, description="Tags for the note")


class NoteAddResult(BaseModel):
    """Result of adding a note."""
    success: bool
    message: str
    file_path: str | None = None


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


@register
def note_add(args: NoteAddArgs) -> NoteAddResult:
    """Add a new note with markdown and frontmatter."""
    ctx = get_tool_context()
    init_database(ctx)
    
    notes_dir = ctx.ensure_notes_dir()
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
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
        
        # Store metadata in database
        content_preview = args.content[:200].replace('\n', ' ')
        cursor.execute(
            """INSERT INTO notes (file_path, title, tags, content_preview) 
               VALUES (?, ?, ?, ?)""",
            (
                str(file_path),
                title,
                json_encode(tags),
                content_preview
            )
        )
        conn.commit()
        
        return NoteAddResult(
            success=True, 
            message=f"Created note: {title}",
            file_path=str(file_path)
        )
        
    except Exception as e:
        conn.rollback()
        return NoteAddResult(success=False, message=f"Error: {str(e)}")
    finally:
        conn.close()


# Search Notes Tool
class NoteSearchArgs(BaseModel):
    """Arguments for searching notes."""
    query: str = Field(description="Search query")


class NoteSearchResult(BaseModel):
    """Result of note search."""
    success: bool
    notes: list[Note] = Field(default_factory=list)
    count: int = 0


@register
def note_search(args: NoteSearchArgs) -> NoteSearchResult:
    """Search notes by content and metadata."""
    ctx = get_tool_context()
    init_database(ctx)
    
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Search in database metadata
        cursor.execute(
            """SELECT * FROM notes 
               WHERE title LIKE ? OR content_preview LIKE ? OR tags LIKE ?
               ORDER BY updated_at DESC""",
            (f"%{args.query}%", f"%{args.query}%", f"%{args.query}%")
        )
        
        rows = cursor.fetchall()
        notes_list = []
        
        for row in rows:
            row_dict = row_to_dict(row)
            # Also search in actual file content for better results
            file_path = Path(row_dict["file_path"])
            if file_path.exists():
                content = file_path.read_text()
                if args.query.lower() in content.lower():
                    notes_list.append(Note(
                        file_path=str(file_path),
                        title=row_dict["title"],
                        tags=json_decode(row_dict["tags"]),
                        content_preview=row_dict["content_preview"],
                        created_at=row_dict["created_at"],
                        updated_at=row_dict["updated_at"]
                    ))
                    
        return NoteSearchResult(success=True, notes=notes_list, count=len(notes_list))
        
    except Exception as e:
        return NoteSearchResult(success=False)
    finally:
        conn.close()


# List Notes Tool
class NoteListArgs(BaseModel):
    """Arguments for listing notes."""
    tags: list[str] = Field(default_factory=list, description="Filter by tags")
    limit: int = Field(50, description="Maximum notes to return")


class NoteListResult(BaseModel):
    """Result of listing notes."""
    success: bool
    notes: list[Note] = Field(default_factory=list)
    count: int = 0


@register
def note_list(args: NoteListArgs) -> NoteListResult:
    """List all notes, optionally filtered by tags."""
    ctx = get_tool_context()
    init_database(ctx)
    
    conn = ctx.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # List all notes, optionally filtered by tags
        query = "SELECT * FROM notes"
        params = []
        
        if args.tags:
            # Filter by any of the provided tags
            tag_conditions = []
            for tag in args.tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            query += " WHERE " + " OR ".join(tag_conditions)
            
        query += f" ORDER BY updated_at DESC LIMIT {args.limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        notes_list = []
        for row in rows:
            row_dict = row_to_dict(row)
            file_path = Path(row_dict["file_path"])
            if file_path.exists():  # Only include if file still exists
                notes_list.append(Note(
                    file_path=str(file_path),
                    title=row_dict["title"],
                    tags=json_decode(row_dict["tags"]),
                    content_preview=row_dict["content_preview"],
                    created_at=row_dict["created_at"],
                    updated_at=row_dict["updated_at"]
                ))
                
        return NoteListResult(success=True, notes=notes_list, count=len(notes_list))
        
    except Exception as e:
        return NoteListResult(success=False)
    finally:
        conn.close()


if __name__ == "__main__":
    simple_cli_main([note_add, note_search, note_list], default_function="note_list")
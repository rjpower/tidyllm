"""Tests for notes tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.notes import (
    NoteAddArgs,
    note_add,
    note_list,
    note_search,
)


@pytest.fixture
def test_context():
    """Create a test context with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            config_dir=temp_path,
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


def test_notes_add_basic(test_context):
    """Test adding a basic note."""
    args = NoteAddArgs(
        title="Test Note",
        content="This is a test note with some content.",
        tags=["test", "example"]
    )
    
    with set_tool_context(test_context):
        file_path_str = note_add(args)
    
    # Function completed successfully if no exception raised
    assert file_path_str is not None
    
    # Verify file was created
    file_path = Path(file_path_str)
    assert file_path.exists()
    
    # Verify content
    content = file_path.read_text()
    assert "Test Note" in content
    assert "This is a test note" in content
    assert "test" in content
    assert "example" in content


def test_notes_add_without_title(test_context):
    """Test adding a note without explicit title."""
    args = NoteAddArgs(
        content="Note without title"
    )
    
    with set_tool_context(test_context):
        file_path_str = note_add(args)
    
    # Function completed successfully if no exception raised
    assert file_path_str is not None
    
    # Should generate a default title
    file_path = Path(file_path_str)
    assert file_path.exists()


def test_notes_list_empty(test_context):
    """Test listing notes in empty directory."""
    with set_tool_context(test_context):
        result = note_list()
    
    # Function completed successfully if no exception raised
    assert list(result) == []


def test_notes_list_with_notes(test_context):
    """Test listing notes with existing notes."""
    # Add some notes first
    with set_tool_context(test_context):
        note_add(NoteAddArgs(title="Note 1", content="Content 1", tags=["tag1"]))
        note_add(NoteAddArgs(title="Note 2", content="Content 2", tags=["tag2"]))
        note_add(NoteAddArgs(title="Note 3", content="Content 3", tags=["tag1", "tag3"]))
        
        # List all notes
        result = note_list()
    
    # Function completed successfully if no exception raised
    notes = list(result)
    assert len(notes) == 3
    
    # Verify note structure
    note = notes[0]
    assert hasattr(note, 'title')
    assert hasattr(note, 'file_path')
    assert hasattr(note, 'tags')
    assert hasattr(note, 'content_preview')


def test_notes_list_filtered_by_tags(test_context):
    """Test listing notes filtered by tags."""
    # Add notes with different tags
    with set_tool_context(test_context):
        note_add(NoteAddArgs(title="Note 1", content="Content 1", tags=["python"]))
        note_add(NoteAddArgs(title="Note 2", content="Content 2", tags=["javascript"]))
        note_add(NoteAddArgs(title="Note 3", content="Content 3", tags=["python", "tutorial"]))
        
        # Filter by tag
        result = note_list(tags=["python"])
    
    # Function completed successfully if no exception raised
    notes = list(result)
    assert len(notes) == 2
    
    # Verify all returned notes have the python tag
    for note in notes:
        assert "python" in note.tags


def test_notes_search_basic(test_context):
    """Test basic note searching."""
    # Add some notes
    with set_tool_context(test_context):
        note_add(NoteAddArgs(title="Python Tutorial", content="Learn Python programming"))
        note_add(NoteAddArgs(title="JavaScript Guide", content="Learn JavaScript basics"))
        note_add(NoteAddArgs(title="Database Design", content="Python database connections"))
        
        # Search for "Python"
        result = note_search("Python")
    
    # Function completed successfully if no exception raised
    notes = list(result)
    assert len(notes) == 2  # Should find 2 notes containing "Python"


def test_notes_search_no_results(test_context):
    """Test search with no matching results."""
    # Add a note
    with set_tool_context(test_context):
        note_add(NoteAddArgs(title="Test", content="Simple content"))
        
        # Search for something that doesn't exist
        result = note_search("nonexistent")
    
    # Function completed successfully if no exception raised
    notes = list(result)
    assert len(notes) == 0


def test_notes_frontmatter_parsing(test_context):
    """Test that frontmatter is correctly parsed and handled."""
    args = NoteAddArgs(
        title="YAML Test",
        content="This note tests YAML frontmatter",
        tags=["yaml", "frontmatter", "test"]
    )
    
    with set_tool_context(test_context):
        file_path_str = note_add(args)
    # Function completed successfully if no exception raised
    
    # Read the file and verify frontmatter
    file_path = Path(file_path_str)
    content = file_path.read_text()
    
    assert content.startswith("---\n")
    assert "title: \"YAML Test\"" in content
    assert "tags: [" in content
    assert "yaml" in content
    assert "frontmatter" in content
    assert "test" in content


def test_notes_filename_sanitization(test_context):
    """Test that filenames are properly sanitized."""
    args = NoteAddArgs(
        title="Invalid/Filename<>Characters",
        content="Testing filename sanitization"
    )
    
    with set_tool_context(test_context):
        file_path_str = note_add(args)
    # Function completed successfully if no exception raised
    
    file_path = Path(file_path_str)
    # Should not contain invalid characters
    assert "<" not in file_path.name
    assert ">" not in file_path.name
    assert "/" not in file_path.name


def test_notes_duplicate_titles(test_context):
    """Test handling of duplicate note titles."""
    # Add first note
    with set_tool_context(test_context):
        file_path1 = note_add(NoteAddArgs(title="Duplicate", content="First note"))
        # Function completed successfully if no exception raised
        
        # Add second note with same title
        file_path2 = note_add(NoteAddArgs(title="Duplicate", content="Second note"))
    # Function completed successfully if no exception raised
    
    # Should have different file paths
    assert file_path1 != file_path2
    
    # Both files should exist
    assert Path(file_path1).exists()
    assert Path(file_path2).exists()


def test_notes_list_with_limit(test_context):
    """Test listing notes with limit."""
    # Add multiple notes
    with set_tool_context(test_context):
        for i in range(5):
            note_add(NoteAddArgs(title=f"Note {i}", content=f"Content {i}"))
        
        # List with limit
        result = note_list(limit=3)
    
    # Function completed successfully if no exception raised
    notes = list(result)
    assert len(notes) == 3

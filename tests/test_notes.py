"""Tests for notes tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.notes import NotesArgs, notes


@pytest.fixture
def test_context():
    """Create a test context with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            user_db=temp_path / "test.db",
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


def test_notes_add_basic(test_context):
    """Test adding a basic note."""
    args = NotesArgs(
        operation="add",
        title="Test Note",
        content="This is a test note with some content.",
        tags=["test", "example"]
    )
    
    result = notes(args, ctx=test_context)
    
    assert result.success is True
    assert "Test Note" in result.message
    assert result.file_path is not None
    
    # Verify file was created
    file_path = Path(result.file_path)
    assert file_path.exists()
    
    # Verify content
    content = file_path.read_text()
    assert "Test Note" in content
    assert "This is a test note" in content
    assert "test" in content
    assert "example" in content


def test_notes_add_without_title(test_context):
    """Test adding a note without explicit title."""
    args = NotesArgs(
        operation="add",
        content="Note without title"
    )
    
    result = notes(args, ctx=test_context)
    
    assert result.success is True
    assert result.file_path is not None
    
    # Should generate a default title
    file_path = Path(result.file_path)
    assert file_path.exists()


def test_notes_add_no_content(test_context):
    """Test adding a note without content fails."""
    args = NotesArgs(
        operation="add",
        title="Empty Note"
    )
    
    result = notes(args, ctx=test_context)
    
    assert result.success is False
    assert "content is required" in result.message.lower()


def test_notes_list_empty(test_context):
    """Test listing notes in empty directory."""
    args = NotesArgs(operation="list")
    result = notes(args, ctx=test_context)
    
    assert result.success is True
    assert result.notes == []


def test_notes_list_with_notes(test_context):
    """Test listing notes with existing notes."""
    # Add some notes first
    notes(NotesArgs(operation="add", title="Note 1", content="Content 1", tags=["tag1"]), ctx=test_context)
    notes(NotesArgs(operation="add", title="Note 2", content="Content 2", tags=["tag2"]), ctx=test_context)
    notes(NotesArgs(operation="add", title="Note 3", content="Content 3", tags=["tag1", "tag3"]), ctx=test_context)
    
    # List all notes
    result = notes(NotesArgs(operation="list"), ctx=test_context)
    
    assert result.success is True
    assert len(result.notes) == 3
    
    # Verify note structure
    note = result.notes[0]
    assert hasattr(note, 'title')
    assert hasattr(note, 'file_path')
    assert hasattr(note, 'tags')
    assert hasattr(note, 'content_preview')


def test_notes_list_filtered_by_tags(test_context):
    """Test listing notes filtered by tags."""
    # Add notes with different tags
    notes(NotesArgs(operation="add", title="Note 1", content="Content 1", tags=["python"]), ctx=test_context)
    notes(NotesArgs(operation="add", title="Note 2", content="Content 2", tags=["javascript"]), ctx=test_context)
    notes(NotesArgs(operation="add", title="Note 3", content="Content 3", tags=["python", "tutorial"]), ctx=test_context)
    
    # Filter by tag
    result = notes(NotesArgs(operation="list", tags=["python"]), ctx=test_context)
    
    assert result.success is True
    assert len(result.notes) == 2
    
    # Verify all returned notes have the python tag
    for note in result.notes:
        assert "python" in note.tags


def test_notes_search_basic(test_context):
    """Test basic note searching."""
    # Add some notes
    notes(NotesArgs(operation="add", title="Python Tutorial", content="Learn Python programming"), ctx=test_context)
    notes(NotesArgs(operation="add", title="JavaScript Guide", content="Learn JavaScript basics"), ctx=test_context)
    notes(NotesArgs(operation="add", title="Database Design", content="Python database connections"), ctx=test_context)
    
    # Search for "Python"
    result = notes(NotesArgs(operation="search", query="Python"), ctx=test_context)
    
    assert result.success is True
    assert len(result.notes) == 2  # Should find 2 notes containing "Python"


def test_notes_search_no_query(test_context):
    """Test search without query fails."""
    args = NotesArgs(operation="search")
    result = notes(args, ctx=test_context)
    
    assert result.success is False
    assert "query is required" in result.message.lower()


def test_notes_search_no_results(test_context):
    """Test search with no matching results."""
    # Add a note
    notes(NotesArgs(operation="add", title="Test", content="Simple content"), ctx=test_context)
    
    # Search for something that doesn't exist
    result = notes(NotesArgs(operation="search", query="nonexistent"), ctx=test_context)
    
    assert result.success is True
    assert len(result.notes) == 0


def test_notes_frontmatter_parsing(test_context):
    """Test that frontmatter is correctly parsed and handled."""
    args = NotesArgs(
        operation="add",
        title="YAML Test",
        content="This note tests YAML frontmatter",
        tags=["yaml", "frontmatter", "test"]
    )
    
    result = notes(args, ctx=test_context)
    assert result.success is True
    
    # Read the file and verify frontmatter
    file_path = Path(result.file_path)
    content = file_path.read_text()
    
    assert content.startswith("---\n")
    assert "title: \"YAML Test\"" in content
    assert "tags: [" in content
    assert "yaml" in content
    assert "frontmatter" in content
    assert "test" in content


def test_notes_filename_sanitization(test_context):
    """Test that filenames are properly sanitized."""
    args = NotesArgs(
        operation="add",
        title="Invalid/Filename<>Characters",
        content="Testing filename sanitization"
    )
    
    result = notes(args, ctx=test_context)
    assert result.success is True
    
    file_path = Path(result.file_path)
    # Should not contain invalid characters
    assert "<" not in file_path.name
    assert ">" not in file_path.name
    assert "/" not in file_path.name


def test_notes_duplicate_titles(test_context):
    """Test handling of duplicate note titles."""
    # Add first note
    result1 = notes(NotesArgs(operation="add", title="Duplicate", content="First note"), ctx=test_context)
    assert result1.success is True
    
    # Add second note with same title
    result2 = notes(NotesArgs(operation="add", title="Duplicate", content="Second note"), ctx=test_context)
    assert result2.success is True
    
    # Should have different file paths
    assert result1.file_path != result2.file_path
    
    # Both files should exist
    assert Path(result1.file_path).exists()
    assert Path(result2.file_path).exists()


def test_notes_invalid_operation():
    """Test invalid operation handling at Pydantic validation level."""
    with pytest.raises(ValueError, match="Input should be"):
        NotesArgs(
            operation="invalid",  # type: ignore
            content="test"
        )
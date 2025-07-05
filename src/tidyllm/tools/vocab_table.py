"""Vocabulary table management tools."""

from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.database import json_decode, json_encode
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class VocabItem(BaseModel):
    """A vocabulary item."""
    id: int
    word: str
    translation: str
    examples: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: str  # ISO format string
    updated_at: str  # ISO format string


# Add Vocab Word Tool
class VocabAddArgs(BaseModel):
    """Arguments for adding a vocabulary word."""
    word: str = Field(description="Word to add")
    translation: str = Field(description="Translation of the word")
    examples: list[str] = Field(default_factory=list, description="Example sentences")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


@register()
def vocab_add(args: VocabAddArgs) -> None:
    """Add a new vocabulary word to the database.
    
    Example usage: vocab_add({"word": "hello", "translation": "hola", "examples": ["Hello world"], "tags": ["greetings"]})
    """
    ctx = get_tool_context()
    db = ctx.db

    db.mutate(
        """INSERT INTO vocab (word, translation, examples, tags) 
            VALUES (?, ?, ?, ?)""",
        (
            args.word,
            args.translation,
            json_encode(args.examples),
            json_encode(args.tags),
        ),
    )


# Search Vocab Tool
class VocabSearchArgs(BaseModel):
    """Arguments for searching vocabulary."""
    word: str | None = Field(None, description="Search by word (partial match)")
    translation: str | None = Field(None, description="Search by translation (partial match)")
    tag: str | None = Field(None, description="Search by tag")
    limit: int = Field(50, description="Maximum results to return")


class VocabSearchResult(BaseModel):
    """Result of vocabulary search."""
    items: list[VocabItem]
    count: int


@register()
def vocab_search(args: VocabSearchArgs) -> VocabSearchResult:
    """Search vocabulary words in the database.
    
    Example usage: vocab_search({"word": "hel", "limit": 10}) or vocab_search({"tag": "verbs"})
    """
    ctx = get_tool_context()
    db = ctx.db

    # Build query with filters
    where_clauses = []
    params = []

    if args.word:
        where_clauses.append("word LIKE ?")
        params.append(f"%{args.word}%")
    if args.translation:
        where_clauses.append("translation LIKE ?")
        params.append(f"%{args.translation}%")
    if args.tag:
        where_clauses.append("tags LIKE ?")
        params.append(f'%"{args.tag}"%')

    query = "SELECT * FROM vocab"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += f" ORDER BY updated_at DESC LIMIT {args.limit}"

    cursor = db.query(query, params)

    items = []
    for row in cursor:
        items.append(
            VocabItem(
                id=row["id"],
                word=row["word"],
                translation=row["translation"],
                examples=json_decode(row["examples"]),
                tags=json_decode(row["tags"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )

    return VocabSearchResult(items=items, count=len(items))


# Update Vocab Tool
class VocabUpdateArgs(BaseModel):
    """Arguments for updating a vocabulary word."""
    word: str = Field(description="Word to update")
    translation: str | None = Field(None, description="New translation")
    examples: list[str] | None = Field(None, description="New example sentences")
    tags: list[str] | None = Field(None, description="New tags")


@register()
def vocab_update(args: VocabUpdateArgs) -> None:
    """Update an existing vocabulary word.
    
    Example usage: vocab_update({"word": "hello", "translation": "¡hola!", "examples": ["Hello there!"]})
    """
    ctx = get_tool_context()
    db = ctx.db

    # Build update query dynamically
    update_parts = []
    params = []

    if args.translation is not None:
        update_parts.append("translation = ?")
        params.append(args.translation)
    if args.examples is not None:
        update_parts.append("examples = ?")
        params.append(json_encode(args.examples))
    if args.tags is not None:
        update_parts.append("tags = ?")
        params.append(json_encode(args.tags))

    if not update_parts:
        raise ValueError("No fields to update")

    params.append(args.word)
    rowcount = db.mutate(
        f"UPDATE vocab SET {', '.join(update_parts)} WHERE word = ?", params
    )

    if rowcount == 0:
        raise ValueError(f"Word not found: {args.word}")

    # Word updated successfully


@register()
def vocab_delete(word: str) -> None:
    """Delete a vocabulary word from the database.
    
    Args:
        word: Word to delete
        
    Example usage: vocab_delete("hello")
    """
    ctx = get_tool_context()
    db = ctx.db

    rowcount = db.mutate("DELETE FROM vocab WHERE word = ?", (word,))

    if rowcount == 0:
        raise ValueError(f"Word not found: {word}")

    # Word deleted successfully


if __name__ == "__main__":
    cli_main(
        [vocab_add, vocab_search, vocab_update, vocab_delete],
        context_cls=ToolContext,
    )

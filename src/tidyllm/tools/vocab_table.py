"""Vocabulary table management tools."""

from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context

# Table is now an alias for Table
from tidyllm.serialization import to_json_string, from_json_string
from tidyllm.linq import Table
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


@register()
def vocab_add(
    word: str,
    translation: str,
    examples: list[str] | None = None,
    tags: list[str] | None = None,
) -> None:
    """Add a new vocabulary word to the database.
    
    Args:
        word: Word to add
        translation: Translation of the word
        examples: Example sentences (optional)
        tags: Tags for categorization (optional)
    
    Example usage: vocab_add("hello", "hola", ["Hello world"], ["greetings"])
    """
    ctx = get_tool_context()
    db = ctx.db

    if examples is None:
        examples = []
    if tags is None:
        tags = []

    db.mutate(
        """INSERT INTO vocab (word, translation, examples, tags) 
            VALUES (?, ?, ?, ?)""",
        (
            word,
            translation,
            to_json_string(examples),
            to_json_string(tags),
        ),
    )


@register()
def vocab_search(
    word: str | None = None,
    translation: str | None = None,
    tag: str | None = None,
    limit: int = 50,
) -> Table:
    """Search vocabulary words in the database.
    
    Args:
        word: Search by word (partial match)
        translation: Search by translation (partial match)
        tag: Search by tag
        limit: Maximum results to return
    
    Returns:
        Table containing matching vocabulary items
    
    Example usage: vocab_search(word="hel", limit=10) or vocab_search(tag="verbs")
    """
    ctx = get_tool_context()
    db = ctx.db

    # Build query with filters
    where_clauses = []
    params = []

    if word:
        where_clauses.append("word LIKE ?")
        params.append(f"%{word}%")
    if translation:
        where_clauses.append("translation LIKE ?")
        params.append(f"%{translation}%")
    if tag:
        where_clauses.append("tags LIKE ?")
        params.append(f'%"{tag}"%')

    query = "SELECT * FROM vocab"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += f" ORDER BY updated_at DESC LIMIT {limit}"

    cursor = db.query(query, params)

    items = []
    for row in cursor:
        items.append(
            VocabItem(
                id=row["id"],
                word=row["word"],
                translation=row["translation"],
                examples=from_json_string(row["examples"], list) if row["examples"] else [],
                tags=from_json_string(row["tags"], list) if row["tags"] else [],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )

    return Table.from_pydantic(items)


@register()
def vocab_update(
    word: str,
    translation: str | None = None,
    examples: list[str] | None = None,
    tags: list[str] | None = None,
) -> None:
    """Update an existing vocabulary word.
    
    Args:
        word: Word to update
        translation: New translation (optional)
        examples: New example sentences (optional)
        tags: New tags (optional)
    
    Example usage: vocab_update("hello", translation="¡hola!", examples=["Hello there!"])
    """
    ctx = get_tool_context()
    db = ctx.db

    # Build update query dynamically
    update_parts = []
    params = []

    if translation is not None:
        update_parts.append("translation = ?")
        params.append(translation)
    if examples is not None:
        update_parts.append("examples = ?")
        params.append(to_json_string(examples))
    if tags is not None:
        update_parts.append("tags = ?")
        params.append(to_json_string(tags))

    if not update_parts:
        raise ValueError("No fields to update")

    params.append(word)
    rowcount = db.mutate(
        f"UPDATE vocab SET {', '.join(update_parts)} WHERE word = ?", params
    )

    if rowcount == 0:
        raise ValueError(f"Word not found: {word}")


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

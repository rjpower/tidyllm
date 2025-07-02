"""Anki flashcard management tool."""

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import genanki
from pydantic import BaseModel, Field
from unidecode import unidecode

from tidyllm.cli import multi_cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register


def unicase_compare(x, y):
    """Custom collation function for unicase comparison."""
    x_ = unidecode(x).lower()
    y_ = unidecode(y).lower()
    return 1 if x_ > y_ else -1 if x_ < y_ else 0


def setup_anki_connection(anki_db_path):
    """Set up SQLite connection with custom collations for Anki database."""
    conn = sqlite3.connect(str(anki_db_path))
    conn.row_factory = sqlite3.Row
    conn.create_collation("unicase", unicase_compare)
    return conn

# Anki model for vocabulary cards
VOCAB_MODEL_ID = 1607392319  # Fixed ID for consistent model
VOCAB_MODEL = genanki.Model(
    VOCAB_MODEL_ID,
    'TidyLLM Vocabulary Card',
    fields=[
        {'name': 'Word'},
        {'name': 'Translation'},
        {'name': 'Examples'},
        {'name': 'Audio'},
    ],
    templates=[
        {
            'name': 'Word → Translation',
            'qfmt': '''
                <div class="word">{{Word}}</div>
                {{#Audio}}<div class="audio">[sound:{{Audio}}]</div>{{/Audio}}
            ''',
            'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                <div class="translation">{{Translation}}</div>
                {{#Examples}}<div class="examples">{{Examples}}</div>{{/Examples}}
            ''',
        },
        {
            'name': 'Translation → Word',
            'qfmt': '<div class="translation">{{Translation}}</div>',
            'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                <div class="word">{{Word}}</div>
                {{#Audio}}<div class="audio">[sound:{{Audio}}]</div>{{/Audio}}
                {{#Examples}}<div class="examples">{{Examples}}</div>{{/Examples}}
            ''',
        },
    ],
    css='''
        .card {
            font-family: arial;
            font-size: 20px;
            text-align: center;
            color: black;
            background-color: white;
            max-width: 600px;
            margin: 0 auto;
        }
        .word {
            font-size: 28px;
            font-weight: bold;
            color: #1a73e8;
            margin: 20px 0;
        }
        .translation {
            font-size: 24px;
            color: #333;
            margin: 20px 0;
        }
        .examples {
            font-size: 18px;
            color: #666;
            font-style: italic;
            margin-top: 20px;
            text-align: left;
            padding: 0 20px;
        }
        .audio {
            margin: 10px 0;
        }
        hr#answer {
            margin: 20px 0;
            border: 1px solid #ccc;
        }
    '''
)


class AnkiCard(BaseModel):
    """A vocabulary card to add to Anki."""
    source_word: str = Field(description="Word in source language")
    translated_word: str = Field(description="Translation of the word")
    examples: list[str] = Field(default_factory=list, description="Example sentences")
    audio_path: Path | None = Field(None, description="Path to audio file")


class AnkiQueryArgs(BaseModel):
    """Arguments for querying Anki database."""
    query: str = Field(description="Search query to find in note fields")
    limit: int = Field(100, description="Maximum number of cards to return")
    deck_name: str | None = Field(None, description="Optional deck name to filter by")


class AnkiQueryResult(BaseModel):
    """Result of querying Anki database."""
    cards: list[dict[str, Any]]
    query: str
    count: int


class AnkiCreateArgs(BaseModel):
    """Arguments for creating Anki cards."""
    deck_name: str = Field(description="Name of the deck to create/add to")
    cards: list[AnkiCard] = Field(description="Cards to add to the deck")
    output_path: Path | None = Field(None, description="Where to save the .apkg file")


class AnkiCreateResult(BaseModel):
    """Result of creating Anki cards."""
    success: bool
    deck_path: Path
    cards_created: int
    message: str = ""


class AnkiListResult(BaseModel):
    """Result of listing Anki decks."""

    decks: list[dict[str, Any]]
    count: int


@register()
def anki_query(args: AnkiQueryArgs) -> AnkiQueryResult:
    """Search for notes in Anki database by query text.

    Example usage: anki_query({"query": "health", "limit": 50})
    Example with deck filter: anki_query({"query": "health", "deck_name": "Japanese Vocabulary::N5", "limit": 20})
    """
    ctx = get_tool_context()
    anki_db = ctx.find_anki_db()
    if not anki_db:
        return AnkiQueryResult(
            cards=[],
            query=args.query,
            count=0
        )

    conn = setup_anki_connection(anki_db)
    cursor = conn.cursor()

    try:
        # Build query to search in note fields
        sql_query = """
            SELECT n.id, n.flds, n.tags, c.ord, d.name as deck_name
            FROM notes n
            JOIN cards c ON c.nid = n.id
            JOIN decks d ON c.did = d.id
            WHERE n.flds LIKE ?
        """
        params = [f"%{args.query}%"]

        # Optional deck filter
        if args.deck_name:
            search_name = args.deck_name.replace('::', '\x1f')
            sql_query += " AND d.name = ?"
            params.append(search_name)

        sql_query += f" LIMIT {args.limit}"

        cursor.execute(sql_query, params)
        rows = cursor.fetchall()

        cards = []
        for row in rows:
            fields = row["flds"].split('\x1f')  # Anki field separator
            deck_name = row["deck_name"].replace('\x1f', '::')  # Format deck name
            cards.append({
                "id": row["id"],
                "fields": fields,
                "tags": row["tags"].split() if row["tags"] else [],
                "card_type": row["ord"],
                "deck_name": deck_name
            })

        return AnkiQueryResult(
            cards=cards,
            query=args.query,
            count=len(cards)
        )

    except Exception as e:
        print(f"Error: {e}")
        return AnkiQueryResult(
            cards=[],
            query=args.query,
            count=0
        )
    finally:
        conn.close()


@register()
def anki_create(args: AnkiCreateArgs) -> AnkiCreateResult:
    """Create Anki flashcards using genanki.

    Example usage: anki_create({"deck_name": "My Vocab", "cards": [{"source_word": "hello", "translated_word": "hola", "examples": ["Hello world"]}]})
    """
    # Generate deck ID from name
    deck_id = abs(hash(args.deck_name)) % (10 ** 10)
    deck = genanki.Deck(deck_id, args.deck_name)

    media_files = []

    try:
        for card in args.cards:
            # Format examples as bullet points
            examples_text = ""
            if card.examples:
                examples_text = "<br>".join(f"• {ex}" for ex in card.examples)

            # Handle audio file
            audio_filename = None
            if card.audio_path and card.audio_path.exists():
                audio_filename = f"{card.source_word}_{card.audio_path.name}"
                media_files.append(str(card.audio_path))

            # Create note
            note = genanki.Note(
                model=VOCAB_MODEL,
                fields=[
                    card.source_word,
                    card.translated_word,
                    examples_text,
                    audio_filename or ""
                ]
            )
            deck.add_note(note)

        # Determine output path
        if args.output_path:
            output_path = args.output_path
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default to temporary directory
            temp_dir = Path(tempfile.gettempdir())
            output_path = temp_dir / f"{args.deck_name.replace(' ', '_')}.apkg"

        # Create package
        package = genanki.Package(deck)
        package.media_files = media_files
        package.write_to_file(str(output_path))

        return AnkiCreateResult(
            success=True,
            deck_path=output_path,
            cards_created=len(args.cards),
            message=f"Created {len(args.cards)} cards in deck '{args.deck_name}'"
        )

    except Exception as e:
        return AnkiCreateResult(
            success=False,
            deck_path=Path(),
            cards_created=0,
            message=f"Error creating deck: {str(e)}"
        )


@register()
def anki_list() -> AnkiListResult:
    """List all available Anki decks with their card counts (alias for anki_list).

    Example usage: anki_decks()
    """
    ctx = get_tool_context()
    anki_db = ctx.find_anki_db()
    if not anki_db:
        return AnkiListResult(decks=[], count=0)

    conn = setup_anki_connection(anki_db)
    cursor = conn.cursor()

    try:
        # Get all decks with card counts using proper collation
        cursor.execute(
            """
            SELECT d.name as deck_name, 
                   COUNT(c.id) as card_count,
                   d.id as deck_id
            FROM decks d
            LEFT JOIN cards c ON c.did = d.id
            GROUP BY d.id, d.name
            ORDER BY d.name COLLATE unicase
        """
        )
        rows = cursor.fetchall()

        decks = []
        for row in rows:
            deck_name = row["deck_name"].replace(
                "\x1f", "::"
            )  # Replace hierarchy separator with ::
            decks.append(
                {
                    "name": deck_name,
                    "card_count": row["card_count"],
                    "deck_id": row["deck_id"],
                }
            )

        return AnkiListResult(decks=decks, count=len(decks))

    except Exception as e:
        print(f"Error: {e}")
        return AnkiListResult(decks=[], count=0)
    finally:
        conn.close()


if __name__ == "__main__":
    multi_cli_main([anki_list, anki_query, anki_create], default_function="anki_list")

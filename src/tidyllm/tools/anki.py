"""Anki flashcard management tool."""

import hashlib
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import genanki
from pydantic import BaseModel, Field
from unidecode import unidecode

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.linq import Table

# Table is now an alias for Table
from tidyllm.llm import completion_with_schema
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class AnkiListResult(BaseModel):
    """Result of listing Anki decks."""

    decks: list[dict[str, Any]]
    count: int


class AnkiCreateResult(BaseModel):
    """Result of creating Anki cards."""

    deck_path: Path
    cards_created: int
    message: str = ""


class AnkiCard(BaseModel):
    """Represents an Anki card from the database."""

    id: int
    fields: list[str]
    tags: list[str]
    card_type: int
    deck_name: str


class AnkiDeck(BaseModel):
    """Represents an Anki deck."""

    name: str
    card_count: int
    deck_id: int


def unicase_compare(x, y):
    """Custom collation function for unicase comparison."""
    x_ = unidecode(x).lower()
    y_ = unidecode(y).lower()
    return 1 if x_ > y_ else -1 if x_ < y_ else 0


@contextmanager
def setup_anki_connection(anki_db_path):
    """Set up SQLite connection with custom collations for Anki database."""
    conn = sqlite3.connect(str(anki_db_path))
    conn.row_factory = sqlite3.Row
    conn.create_collation("unicase", unicase_compare)
    try:
        yield conn
    finally:
        conn.close()


# Enhanced bilingual vocabulary model with audio support
BILINGUAL_VOCAB_MODEL_ID = 1607392320  # Fixed ID for bilingual model
BILINGUAL_VOCAB_MODEL = genanki.Model(
    BILINGUAL_VOCAB_MODEL_ID,
    "TidyLLM Bilingual Vocabulary Card",
    fields=[
        {"name": "Term"},
        {"name": "Reading"},
        {"name": "Meaning"},
        {"name": "Example"},
        {"name": "ExampleTranslation"},
        {"name": "TermAudio"},
        {"name": "MeaningAudio"},
    ],
    templates=[
        {
            "name": "Term → Meaning",
            "qfmt": """
                <div class="term">{{Term}}</div>
                {{TermAudio}}
                <div class="example">{{Example}}</div>
            """,
            "afmt": """
                {{FrontSide}}
                <hr id="answer">
                {{MeaningAudio}}
                <div class="reading">{{Reading}}</div>
                <div class="meaning">{{Meaning}}</div>
                <div class="example-translation">{{ExampleTranslation}}</div>
            """,
        },
        {
            "name": "Meaning → Term",
            "qfmt": """
                <div class="meaning">{{Meaning}}</div>
                {{MeaningAudio}}
                <div class="example-translation">{{ExampleTranslation}}</div>
            """,
            "afmt": """
                {{FrontSide}}
                <hr id="answer">
                <div class="term">{{Term}}</div>
                {{TermAudio}}
                <div class="reading">{{Reading}}</div>
                <div class="example">{{Example}}</div>
            """,
        },
    ],
    css="""
        .card {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            font-size: 18px;
            text-align: center;
            color: #333;
            background-color: #fafafa;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 8px;
        }
        .term {
            font-size: 32px;
            font-weight: 700;
            color: #2563eb;
            margin: 15px 0;
            line-height: 1.2;
        }
        .reading {
            font-size: 20px;
            color: #7c3aed;
            font-style: italic;
            margin: 10px 0;
        }
        .meaning {
            font-size: 24px;
            color: #059669;
            font-weight: 600;
            margin: 15px 0;
            line-height: 1.3;
        }
        .example {
            font-size: 18px;
            color: #4b5563;
            font-style: italic;
            margin: 15px 0;
            padding: 12px;
            background-color: #f3f4f6;
            border-radius: 6px;
            border-left: 4px solid #6b7280;
        }
        .example-translation {
            font-size: 16px;
            color: #6b7280;
            margin: 10px 0;
            padding: 8px;
            background-color: #e5e7eb;
            border-radius: 4px;
        }
        .audio {
            margin: 12px 0;
        }
        hr#answer {
            margin: 20px 0;
            border: none;
            border-top: 2px solid #e5e7eb;
        }
    """,
)


class AddVocabCardRequest(BaseModel):
    """Request to add a bilingual vocabulary card with audio."""

    term_en: str = Field(description="English term")
    term_ja: str = Field(description="Japanese term")
    reading_ja: str = Field(
        default="", description="Japanese reading (hiragana/katakana)"
    )
    sentence_en: str = Field(description="English example sentence")
    sentence_ja: str = Field(description="Japanese example sentence")
    audio_en: Path | None = Field(None, description="Path to English audio file")
    audio_ja: Path | None = Field(None, description="Path to Japanese audio file")


class AddVocabCardsRequest(BaseModel):
    """Request to add multiple bilingual vocabulary cards with audio."""

    cards: list[AddVocabCardRequest] = Field(description="List of vocabulary cards to add")
    deck_name: str = Field(description="Name of the Anki deck")


class ExampleSentenceResponse(BaseModel):
    """Response model for generated example sentences."""

    source_sentence: str
    target_sentence: str


@register()
def generate_example_sentence(
    word: str,
    translation: str,
    source_language: str = "en",
    target_language: str = "ja",
    difficulty: str = "intermediate",
) -> ExampleSentenceResponse:
    """Generate example sentences for a vocabulary word using LLM.

    Args:
        word: The vocabulary word
        translation: Translation of the word
        source_language: Source language code
        target_language: Target language code
        difficulty: Difficulty level (beginner, intermediate, advanced)

    Returns:
        Tuple of (source_sentence, target_sentence)

    Example usage: generate_example_sentence("hello", "こんにちは", "en", "ja")
    """
    ctx = get_tool_context()

    prompt = f"""Create example sentences for the vocabulary word "{word}" (translation: "{translation}").

Requirements:
- Create one natural example sentence in {source_language} using the word "{word}"
- Translate that sentence to {target_language}
- Use {difficulty} level vocabulary and grammar
- Make the sentences practical and commonly used
- Keep sentences concise (under 20 words)

Return only a JSON object with this format:
{{
    "source_sentence": "The example sentence in {source_language}",
    "target_sentence": "The translation in {target_language}"
}}"""

    return completion_with_schema(
        model=ctx.cfg.fast_model,
        messages=[{"role": "user", "content": prompt}],
        response_schema=ExampleSentenceResponse,
    )


@register()
def anki_add_vocab_cards(
    deck_name: str,
    cards: list[AddVocabCardRequest],
) -> AnkiCreateResult:
    """Create multiple bilingual vocabulary cards with audio and add to Anki deck.

    Args:
        deck_name: Name of the Anki deck
        cards: List of vocabulary cards to add

    Example usage: anki_add_vocab_cards("My Deck", [card1, card2, ...])
    """
    # Generate deck ID from name
    deck_id = abs(hash(deck_name)) % (10**10)
    deck = genanki.Deck(deck_id, deck_name)

    # Create temporary directory for media files
    temp_dir = tempfile.TemporaryDirectory()
    media_files = []

    def _add_audio(audio_path: Path, term: str) -> str:
        """Add audio file to package with content-based filename."""
        if not audio_path or not audio_path.exists():
            return ""

        # Create a unique filename based on content hash
        audio_filename = f"audio_{hashlib.md5(term.encode()).hexdigest()[:8]}.mp3"
        temp_audio_path = Path(temp_dir.name) / audio_filename
        shutil.copy2(audio_path, temp_audio_path)
        media_files.append(temp_audio_path)

        return f"[sound:{audio_filename}]"

    # Process each card
    for card in cards:
        # Handle audio files
        term_audio_field = _add_audio(card.audio_ja, card.term_ja)
        meaning_audio_field = _add_audio(card.audio_en, card.term_en)

        # Create bilingual note using the enhanced model
        note = genanki.Note(
            model=BILINGUAL_VOCAB_MODEL,
            fields=[
                card.term_ja,  # Term
                card.reading_ja,  # Reading
                card.term_en,  # Meaning
                card.sentence_ja,  # Example
                card.sentence_en,  # ExampleTranslation
                term_audio_field,  # TermAudio
                meaning_audio_field,  # MeaningAudio
            ],
        )
        deck.add_note(note)

    # Determine output path in temp directory
    output_temp_dir = Path(tempfile.gettempdir())
    output_path = output_temp_dir / f"{deck_name.replace(' ', '_')}_vocab.apkg"

    # Create package with media files
    package = genanki.Package(deck)
    if media_files:
        package.media_files = media_files

    try:
        package.write_to_file(str(output_path))

        return AnkiCreateResult(
            deck_path=output_path,
            cards_created=len(cards),
            message=f"Created {len(cards)} bilingual vocab cards in deck '{deck_name}'",
        )
    finally:
        # Clean up temporary directory
        temp_dir.cleanup()


@register()
def anki_add_vocab_card(req: AddVocabCardRequest) -> AnkiCreateResult:
    """Create a bilingual vocabulary card with audio and add to Anki deck.

    Example usage: anki_add_vocab_card(AddVocabCardRequest(...))
    """
    # Use the batch function with a single card
    return anki_add_vocab_cards(deck_name="Japanese Vocabulary", cards=[req])


@register()
def anki_query(query: str, limit: int = 100, deck_name: str | None = None) -> Table:
    """Search for notes in Anki database by query text.

    Args:
        query: Search query to find in note fields
        limit: Maximum number of cards to return (default: 100)
        deck_name: Optional deck name to filter by

    Example usage: anki_query("health", 50)
    Example with deck filter: anki_query("health", 20, "Japanese Vocabulary::N5")
    """
    ctx = get_tool_context()
    anki_db = ctx.config.find_anki_db()
    if not anki_db:
        return Table.empty()

    with setup_anki_connection(anki_db) as conn:
        # Build query to search in note fields
        sql_query = """
          SELECT n.id, n.flds, n.tags, c.ord, d.name as deck_name
          FROM notes n
          JOIN cards c ON c.nid = n.id
          JOIN decks d ON c.did = d.id
          WHERE n.flds LIKE ?
      """
        params = [f"%{query}%"]

        # Optional deck filter
        if deck_name:
            search_name = deck_name.replace("::", "\x1f")
            sql_query += " AND d.name = ?"
            params.append(search_name)

        sql_query += f" LIMIT {limit}"

        cursor = conn.execute(sql_query, params)
        rows = cursor.fetchall()

        cards = []
        for row in rows:
            fields = row["flds"].split("\x1f")  # Anki field separator
            deck_name = row["deck_name"].replace("\x1f", "::")  # Format deck name
            cards.append(
                AnkiCard(
                    id=row["id"],
                    fields=fields,
                    tags=row["tags"].split() if row["tags"] else [],
                    card_type=row["ord"],
                    deck_name=deck_name,
                )
            )
        return Table.from_pydantic(cards)


@register()
def anki_list() -> Table:
    """List all available Anki decks with their card counts (alias for anki_list).

    Example usage: anki_decks()
    """
    ctx = get_tool_context()
    anki_db = ctx.config.find_anki_db()
    if not anki_db:
        return Table.empty()

    with setup_anki_connection(anki_db) as conn:
        cursor = conn.execute(
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
                AnkiDeck(
                    name=deck_name,
                    card_count=row["card_count"],
                    deck_id=row["deck_id"],
                )
            )

    return Table.from_pydantic(decks)


if __name__ == "__main__":
    cli_main(
        [
            anki_list,
            anki_query,
            anki_add_vocab_card,
            anki_add_vocab_cards,
            generate_example_sentence,
        ],
        context_cls=ToolContext,
    )

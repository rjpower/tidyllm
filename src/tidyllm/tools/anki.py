"""Anki flashcard management tool."""

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import genanki
import litellm
from pydantic import BaseModel, Field
from unidecode import unidecode

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


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

# Anki model for basic vocabulary cards
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
                {{#Reading}}<div class="reading">{{Reading}}</div>{{/Reading}}
                {{#TermAudio}}<div class="audio">[sound:{{TermAudio}}]</div>{{/TermAudio}}
            """,
            "afmt": """
                {{FrontSide}}
                <hr id="answer">
                <div class="meaning">{{Meaning}}</div>
                {{#MeaningAudio}}<div class="audio">[sound:{{MeaningAudio}}]</div>{{/MeaningAudio}}
                {{#Example}}<div class="example">{{Example}}</div>{{/Example}}
                {{#ExampleTranslation}}<div class="example-translation">{{ExampleTranslation}}</div>{{/ExampleTranslation}}
            """,
        },
        {
            "name": "Meaning → Term",
            "qfmt": """
                <div class="meaning">{{Meaning}}</div>
                {{#MeaningAudio}}<div class="audio">[sound:{{MeaningAudio}}]</div>{{/MeaningAudio}}
            """,
            "afmt": """
                {{FrontSide}}
                <hr id="answer">
                <div class="term">{{Term}}</div>
                {{#Reading}}<div class="reading">{{Reading}}</div>{{/Reading}}
                {{#TermAudio}}<div class="audio">[sound:{{TermAudio}}]</div>{{/TermAudio}}
                {{#Example}}<div class="example">{{Example}}</div>{{/Example}}
                {{#ExampleTranslation}}<div class="example-translation">{{ExampleTranslation}}</div>{{/ExampleTranslation}}
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


class AnkiCard(BaseModel):
    """A vocabulary card to add to Anki."""
    source_word: str = Field(description="Word in source language")
    translated_word: str = Field(description="Translation of the word")
    examples: list[str] = Field(default_factory=list, description="Example sentences")
    audio_path: Path | None = Field(None, description="Path to audio file")


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
    deck_name: str = Field(description="Name of the Anki deck")


@register()
def generate_example_sentence(
    word: str,
    translation: str,
    source_language: str = "en",
    target_language: str = "ja",
    difficulty: str = "intermediate",
) -> tuple[str, str]:
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

    response = litellm.completion(
        model=ctx.config.fast_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "example_sentences",
                "schema": {
                    "type": "object",
                    "properties": {
                        "source_sentence": {"type": "string"},
                        "target_sentence": {"type": "string"},
                    },
                    "required": ["source_sentence", "target_sentence"],
                },
                "strict": True,
            },
        },
    )

    if response.choices:
        import json

        result = json.loads(response.choices[0].message.content)
        return result["source_sentence"], result["target_sentence"]
    else:
        # Fallback sentences
        return f"This is an example with {word}.", f"これは{translation}の例です。"


@register()
def anki_add_vocab_card(req: AddVocabCardRequest) -> AnkiCreateResult:
    """Create a bilingual vocabulary card with audio and add to Anki deck.

    Args:
        req: Request containing all card data

    Returns:
        AnkiCreateResult with creation details

    Example usage: anki_add_vocab_card(AddVocabCardRequest(...))
    """
    # Generate deck ID from name
    deck_id = abs(hash(req.deck_name)) % (10**10)
    deck = genanki.Deck(deck_id, req.deck_name)

    media_files = []

    # Handle audio files
    term_audio_filename = None
    if req.audio_ja and req.audio_ja.exists():
        term_audio_filename = f"{req.term_ja}_term.mp3"
        media_files.append(str(req.audio_ja))

    cli_mainfilename = None
    if req.audio_en and req.audio_en.exists():
        meaning_audio_filename = f"{req.term_en}_meaning.mp3"
        media_files.append(str(req.audio_en))

    # Create bilingual note using the enhanced model
    note = genanki.Note(
        model=BILINGUAL_VOCAB_MODEL,
        fields=[
            req.term_ja,  # Term
            req.reading_ja,  # Reading
            req.term_en,  # Meaning
            req.sentence_ja,  # Example
            req.sentence_en,  # ExampleTranslation
            term_audio_filename or "",  # TermAudio
            meaning_audio_filename or "",  # MeaningAudio
        ],
    )
    deck.add_note(note)

    # Determine output path in temp directory
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / f"{req.deck_name.replace(' ', '_')}_vocab.apkg"

    # Create package with media files
    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(str(output_path))

    return AnkiCreateResult(
        deck_path=output_path,
        cards_created=1,
        message=f"Created bilingual vocab card for '{req.term_ja}' / '{req.term_en}' in deck '{req.deck_name}'",
    )


class AnkiQueryResult(BaseModel):
    """Result of querying Anki database."""
    cards: list[dict[str, Any]]
    query: str
    count: int


class AnkiCreateResult(BaseModel):
    """Result of creating Anki cards."""
    deck_path: Path
    cards_created: int
    message: str = ""


class AnkiListResult(BaseModel):
    """Result of listing Anki decks."""

    decks: list[dict[str, Any]]
    count: int


@register()
def anki_query(query: str, limit: int = 100, deck_name: str | None = None) -> AnkiQueryResult:
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
        return AnkiQueryResult(
            cards=[],
            query=query,
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
        params = [f"%{query}%"]

        # Optional deck filter
        if deck_name:
            search_name = deck_name.replace('::', '\x1f')
            sql_query += " AND d.name = ?"
            params.append(search_name)

        sql_query += f" LIMIT {limit}"

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
            query=query,
            count=len(cards)
        )

    except Exception as e:
        print(f"Error: {e}")
        return AnkiQueryResult(
            cards=[],
            query=query,
            count=0
        )
    finally:
        conn.close()


@register()
def anki_create(deck_name: str, cards: list[AnkiCard], output_path: Path | None = None) -> AnkiCreateResult:
    """Create Anki flashcards using genanki.

    Args:
        deck_name: Name of the deck to create/add to
        cards: Cards to add to the deck
        output_path: Where to save the .apkg file (optional)

    Example usage: anki_create("My Vocab", [AnkiCard(source_word="hello", translated_word="hola", examples=["Hello world"])])
    """
    # Generate deck ID from name
    deck_id = abs(hash(deck_name)) % (10 ** 10)
    deck = genanki.Deck(deck_id, deck_name)

    media_files = []

    for card in cards:
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
    if output_path:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Default to temporary directory
        temp_dir = Path(tempfile.gettempdir())
        output_path = temp_dir / f"{deck_name.replace(' ', '_')}.apkg"

    # Create package
    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(str(output_path))

    return AnkiCreateResult(
        deck_path=output_path,
        cards_created=len(cards),
        message=f"Created {len(cards)} cards in deck '{deck_name}'"
    )


@register()
def anki_list() -> AnkiListResult:
    """List all available Anki decks with their card counts (alias for anki_list).

    Example usage: anki_decks()
    """
    ctx = get_tool_context()
    anki_db = ctx.config.find_anki_db()
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
    multi_cli_main(
        [
            anki_list,
            anki_query,
            anki_create,
            anki_add_vocab_card,
            generate_example_sentence,
        ],
        default_function="anki_list",
        context_cls=ToolContext,
    )

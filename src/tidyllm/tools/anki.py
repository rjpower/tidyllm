"""Anki flashcard management tool."""

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import genanki
from pydantic import BaseModel, Field

from tidyllm.multi_cli import simple_cli_main
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext

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


class AnkiReadArgs(BaseModel):
    """Arguments for reading from Anki database."""
    deck_name: str = Field(description="Name of the deck to read from")
    limit: int = Field(100, description="Maximum number of cards to return")
    tags: list[str] | None = Field(None, description="Filter by tags")


class AnkiReadResult(BaseModel):
    """Result of reading from Anki database."""
    cards: list[dict[str, Any]]
    deck_name: str
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
    message: str | None = None


@register
def anki_read(args: AnkiReadArgs, *, ctx: ToolContext) -> AnkiReadResult:
    """Read vocabulary items from Anki database."""
    anki_db = ctx.find_anki_db()
    if not anki_db:
        return AnkiReadResult(
            cards=[],
            deck_name=args.deck_name,
            count=0
        )
    
    conn = sqlite3.connect(str(anki_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get deck ID
        cursor.execute(
            "SELECT id FROM decks WHERE name = ?",
            (args.deck_name,)
        )
        deck_row = cursor.fetchone()
        
        if not deck_row:
            return AnkiReadResult(
                cards=[],
                deck_name=args.deck_name,
                count=0
            )
            
        deck_id = deck_row["id"]
        
        # Query cards from the deck
        query = """
            SELECT n.id, n.flds, n.tags, c.ord
            FROM notes n
            JOIN cards c ON c.nid = n.id
            WHERE c.did = ?
        """
        params = [deck_id]
        
        if args.tags:
            tag_conditions = []
            for tag in args.tags:
                tag_conditions.append("n.tags LIKE ?")
                params.append(f"%{tag}%")
            query += " AND (" + " OR ".join(tag_conditions) + ")"
            
        query += f" LIMIT {args.limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        cards = []
        for row in rows:
            fields = row["flds"].split('\x1f')  # Anki field separator
            cards.append({
                "id": row["id"],
                "fields": fields,
                "tags": row["tags"].split() if row["tags"] else [],
                "card_type": row["ord"]
            })
            
        return AnkiReadResult(
            cards=cards,
            deck_name=args.deck_name,
            count=len(cards)
        )
        
    except Exception:
        return AnkiReadResult(
            cards=[],
            deck_name=args.deck_name,
            count=0
        )
    finally:
        conn.close()


@register
def anki_create(args: AnkiCreateArgs, *, ctx: ToolContext) -> AnkiCreateResult:
    """Create Anki flashcards using genanki."""
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


if __name__ == "__main__":
    simple_cli_main([anki_read, anki_create], default_function="anki_read")
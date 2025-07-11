#!/usr/bin/env python3
"""Anki flashcard creation app with TTS audio generation.

This app creates bilingual vocabulary flashcards with generated example sentences
and TTS audio for both languages, then creates Anki packages for import.
"""

import csv
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
from tidyllm.llm import completion_with_schema
from tidyllm.registry import register
from tidyllm.tools.anki import (
    AddVocabCardRequest,
    anki_add_vocab_cards,
)
from tidyllm.tools.context import ToolContext
from tidyllm.tools.tts import generate_speech
from tidyllm.types.linq import Table
from tidyllm.ui.selection import select_ui

console = Console()


class AddCardRequest(BaseModel):
    """Request to add a single vocabulary card."""

    term_en: str = ""
    term_ja: str = ""
    reading_ja: str = ""
    sentence_en: str = ""
    sentence_ja: str = ""


class AddCardResult(BaseModel):
    """Result of adding a single card."""

    card_created: bool
    deck_path: Path
    message: str


class BatchAddResult(BaseModel):
    """Result of batch adding cards."""

    cards_created: int
    total_cards: int
    deck_path: Path
    failed_cards: list[str] = []


class VocabularyItem(BaseModel):
    """A single vocabulary item extracted from text."""

    word: str
    translation: str
    reading: str


@cached_function
def _infer_missing_fields(request: AddCardRequest) -> AddCardRequest:
    """Infer missing fields for a vocabulary card using LLM.

    Args:
        request: AddCardRequest with potentially missing fields

    Returns:
        AddCardRequest with all fields completed
    """
    ctx = get_tool_context()

    # Convert request to dict for processing, excluding non-serializable fields
    request_data = request.model_dump(exclude_unset=False, exclude={"output_dir"})

    prompt = f"""
<instructions>
Given the vocabulary item in the <input> section below, infer any missing fields.
Description of each field:

- term_en: the English term/meaning
- term_ja: the Japanese term (can be in kanji, hiragana, or katakana)
- reading_ja: the phonetic reading of the Japanese term in hiragana or katakana
- sentence_en: a natural English sentence using the English term in context
- sentence_ja: a natural Japanese sentence using the Japanese term in context, with Ruby annotations for pronunciation

Output only JSON.
Output a single JSON object with the completed vocabulary item.

{{
  "term_en": "library",
  "term_ja": "図書館",
  "reading_ja": "としょかん",
  "sentence_en": "I borrowed a book from the library.",
  "sentence_ja": "<ruby>図書館<rt>としょかん</rt></ruby>から<ruby>本<rt>ほん</rt></ruby>を<ruby>借<rt>か</rt></ruby>りました。"
}}
</instructions>

<input>
{json.dumps(request_data, ensure_ascii=False, sort_keys=True)}
</input>
"""

    return completion_with_schema(
        model=ctx.config.fast_model,
        messages=[{"role": "user", "content": prompt}],
        response_schema=AddCardRequest,
    )


class VocabularyExtractionResponse(BaseModel):
    """Response model for vocabulary extraction from text."""
    items: list[VocabularyItem]


def _parse_csv_row(row: dict) -> AddCardRequest:
    """Parse a single CSV row into AddCardRequest."""
    term_en = row.get("term_en", "").strip()
    term_ja = row.get("term_ja", "").strip()
    reading_ja = row.get("reading_ja", "").strip()
    sentence_en = row.get("sentence_en", "").strip()
    sentence_ja = row.get("sentence_ja", "").strip()

    # At least one term (English or Japanese) must be provided
    if not term_en and not term_ja:
        raise ValueError(
            f"Row {row}. Missing required term (either term_en or term_ja)"
        )

    return AddCardRequest(
        term_en=term_en,
        term_ja=term_ja,
        reading_ja=reading_ja,
        sentence_en=sentence_en,
        sentence_ja=sentence_ja,
    )


def _vocab_item_to_card_request(
    item: VocabularyItem, source_language: str, target_language: str
) -> AddCardRequest:
    """Convert VocabularyItem to AddCardRequest."""
    if source_language == "ja" and target_language == "en":
        term_ja = item.word
        term_en = item.translation
        reading_ja = item.reading
    else:
        term_en = item.word
        term_ja = item.translation
        reading_ja = item.reading

    return AddCardRequest(
        term_en=term_en,
        term_ja=term_ja,
        reading_ja=reading_ja,
    )


def _generate_audio_files(
    term_en: str, term_ja: str, output_dir: Path
) -> tuple[Path, Path]:
    """Generate TTS audio files for English and Japanese content.

    Args:
        term_en: English term for filename
        term_ja: Japanese term for filename
        sentence_en: English sentence content for TTS
        sentence_ja: Japanese sentence content for TTS
        output_dir: Directory to save audio files

    Returns:
        Tuple of (english_audio_path, japanese_audio_path)
    """
    # Generate English audio
    audio_en_path = output_dir / f"{term_en.replace(' ', '_')}_en.mp3"
    en_result = list(generate_speech(content=term_en, language="English"))[0]
    with open(audio_en_path, "wb") as f:
        f.write(en_result.to_bytes(format="mp3"))

    # Generate Japanese audio
    audio_ja_path = output_dir / f"{term_ja.replace(' ', '_')}_ja.mp3"
    ja_result = list(generate_speech(content=term_ja, language="Japanese"))[0]
    with open(audio_ja_path, "wb") as f:
        f.write(ja_result.to_bytes(format="mp3"))

    return audio_en_path, audio_ja_path


@register()
def add_card(
    term_en: str = "",
    term_ja: str = "",
    deck_name: str = "Japanese Vocabulary",
):
    """Add a vocabulary card with auto-generated content.

    All fields except the source term are automatically inferred using LLM.

    Args:
        term_en: English term (either term_en or term_ja must be provided)
        term_ja: Japanese term (either term_en or term_ja must be provided)
        deck_name: Name of the Anki deck

    Returns:
        None

    Example: add_card(term_ja="こんにちは")
    Example: add_card(term_en="hello", deck_name="English::Basic")
    """
    # Validate that at least one term is provided
    if not term_en and not term_ja:
        raise ValueError("Either term_en or term_ja must be provided")

    term_display = term_ja if term_ja else term_en
    console.print(f"[bold blue]Creating card for:[/bold blue] {term_display}")

    # Setup output directory
    output_dir = Path(tempfile.mkdtemp(prefix="add_card_"))
    # Create request with provided terms
    request = AddCardRequest(
        term_en=term_en,
        term_ja=term_ja,
    )

    console.print("[yellow]Inferring missing fields...[/yellow]")
    complete_request = _infer_missing_fields(request)

    console.print("[yellow]Generating TTS audio...[/yellow]")
    audio_en_path, audio_ja_path = _generate_audio_files(
        complete_request.term_en, complete_request.term_ja, output_dir
    )

    console.print(f"Generated audio files: EN: {audio_en_path}, JA: {audio_ja_path}")
    console.print("[yellow]Creating Anki card...[/yellow]")

    vocab_card = AddVocabCardRequest(
        term_en=complete_request.term_en,
        term_ja=complete_request.term_ja,
        reading_ja=complete_request.reading_ja,
        sentence_en=complete_request.sentence_en,
        sentence_ja=complete_request.sentence_ja,
        audio_en=audio_en_path,
        audio_ja=audio_ja_path,
    )

    result = anki_add_vocab_cards(deck_name, [vocab_card])

    print("Card created successfully!")
    print(f"Deck file: {result.deck_path}")


@register()
def add_from_csv(
    csv_path: Path,
    deck_name: str = "Japanese Vocabulary",
    interactive: bool = True,
):
    """Add vocabulary cards from CSV file.

    Expected CSV format (only one term per row required, others are optional):
    term_en,term_ja,reading_ja,sentence_en,sentence_ja
    hello,こんにちは,こんにちは,,
    ,ありがとう,ありがとう,,

    Missing fields will be automatically inferred by the LLM.

    Args:
        csv_path: Path to CSV file
        deck_name: Name of the Anki deck
        interactive: Whether to show interactive selection (default: True)

    Returns:
        None

    Example: add_from_csv(Path("vocab.csv"), "Japanese::N5")
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    console.print(f"[bold blue]Processing CSV file:[/bold blue] {csv_path}")

    # Read CSV file and parse using enhanced LINQ operations
    with open(csv_path, encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    # Parse all rows with error handling using LINQ
    successful_results, failed_results = Table.from_rows(csv_rows).try_select(
        _parse_csv_row
    )

    card_requests = successful_results.materialize()
    failed_cards = [str(exception) for exception in failed_results]

    console.print(f"[green]Found {len(card_requests)} valid cards to process[/green]")

    for failure in failed_cards:
        console.print(f"  [yellow]- {failure}[/yellow]")

    review_and_add(card_requests, deck_name, interactive=interactive)


@register()
def add_from_text(
    text: str,
    source_language: str = "ja",
    target_language: str = "en",
    deck_name: str = "Japanese Vocabulary",
    max_words: int = 10,
    output_dir: Path | None = None,
    interactive: bool = True,
):
    """Extract vocabulary from text and create cards.

    Args:
        text: Text to extract vocabulary from
        source_language: Source language of the text
        target_language: Target language for translations
        deck_name: Name of the Anki deck
        max_words: Maximum number of words to extract
        output_dir: Output directory for files
        interactive: Whether to show interactive selection (default: True)

    Returns:
        None

    Example: add_from_text("日本語の文章です", "ja", "en", "Japanese::Reading")
    """
    console.print("[bold blue]Extracting vocabulary from text[/bold blue]")
    console.print(f"[blue]Text:[/blue] {text[:100]}{'...' if len(text) > 100 else ''}")

    # Use LLM to extract vocabulary
    ctx = get_tool_context()

    prompt = f"""Extract the most useful vocabulary words from this {source_language} text for language learners.

Text: "{text}"

Requirements:
- Extract up to {max_words} key vocabulary words
- Focus on useful, common words that learners should know
- Provide accurate translations to {target_language}
- Include pronunciation/reading for Japanese words when applicable
- Skip very basic words (particles, common verbs like "is", "have")

Return a JSON object with vocabulary items:
{{
    "items": [
        {{
            "word": "word in source language",
            "translation": "word in target language", 
            "reading": "pronunciation/reading if applicable"
        }}
    ]
}}"""

    extraction_result = completion_with_schema(
        model=ctx.config.fast_model,
        messages=[{"role": "user", "content": prompt}],
        response_schema=VocabularyExtractionResponse,
    )

    console.print(
        f"[green]Extracted {len(extraction_result.items)} vocabulary words[/green]"
    )

    # Convert vocabulary items to Table[AddCardRequest] using LINQ
    card_requests = (
        Table.from_rows(extraction_result.items)
        .select(
            lambda item: _vocab_item_to_card_request(
                item, source_language, target_language
            )
        )
        .to_list()
    )

    review_and_add(
        Table.from_rows(card_requests),
        deck_name,
        output_dir,
        interactive=interactive,
    )


@register()
def review_and_add(
    card_requests: Table[AddCardRequest],
    deck_name: str = "Japanese Vocabulary",
    output_dir: Path | None = None,
    interactive: bool = True,
):
    """Interactive review and selection of card requests before adding cards.

    Args:
        card_requests: Table of AddCardRequest objects to review
        deck_name: Name of the Anki deck
        output_dir: Output directory for files
        interactive: Whether to show interactive selection (default: True)

    Returns:
        None

    Example: review_and_add(card_requests_table, "Japanese::N5")
    """
    if len(card_requests) == 0:
        console.print("[yellow]No card requests to process[/yellow]")
        return

    if interactive:
        card_requests = select_ui(
            card_requests,
            title="Card Requests - Select cards to create",
            display_columns=["term_en", "term_ja", "reading_ja"],
        )

    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_review_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_audio(req: AddCardRequest):
        return _generate_audio_files(req.term_en, req.term_ja, output_dir=output_dir)

    cards, failed_results = (
        card_requests.with_progress("Processing selected cards")
        .select(_infer_missing_fields)
        .try_select(_generate_audio)
    )

    failed_cards = [f"Processing error: {str(exception)}" for exception in failed_results]

    result = anki_add_vocab_cards(deck_name, cards.materialize())
    cards_created = result.cards_created

    console.print(f"[green]Cards created: {cards_created}/{len(card_requests)}[/green]")
    for failure in failed_cards:
        console.print(f"  [red]- {failure}[/red]")


if __name__ == "__main__":
    functions = [add_card, add_from_csv, add_from_text, review_and_add]
    cli_main(
        functions,
        context_cls=ToolContext,
    )

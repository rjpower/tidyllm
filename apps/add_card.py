#!/usr/bin/env python3
"""Anki flashcard creation app with TTS audio generation.

This app creates bilingual vocabulary flashcards with generated example sentences
and TTS audio for both languages, then creates Anki packages for import.
"""

import csv
import json
import tempfile
from pathlib import Path
from typing import cast

import litellm
from litellm.types.utils import ModelResponse
from pydantic import BaseModel
from rich.console import Console

# Import removed - using LINQ with_progress instead
from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
from tidyllm.linq import Table, from_iterable
from tidyllm.registry import register
from tidyllm.tools.anki import (
    AddVocabCardRequest,
    anki_add_vocab_cards,
)
from tidyllm.tools.context import ToolContext
from tidyllm.tools.tts import generate_speech
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

    response = cast(
        ModelResponse,
        litellm.completion(
            model=ctx.config.fast_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=AddCardRequest,
        ),
    )

    message_content = cast(litellm.Choices, response.choices)[0].message.content
    assert message_content is not None, "Response content is None"
    return AddCardRequest.model_validate_json(message_content)


class VocabularyExtractionResponse(BaseModel):
    """Response model for vocabulary extraction from text."""
    items: list[VocabularyItem]


def _parse_csv_row(row: dict, deck_name: str, row_index: int) -> AddCardRequest:
    """Parse a single CSV row into AddCardRequest."""
    term_en = row.get("term_en", "").strip()
    term_ja = row.get("term_ja", "").strip()
    reading_ja = row.get("reading_ja", "").strip()
    sentence_en = row.get("sentence_en", "").strip()
    sentence_ja = row.get("sentence_ja", "").strip()

    # At least one term (English or Japanese) must be provided
    if not term_en and not term_ja:
        raise ValueError(f"Row {row_index}: Missing required term (either term_en or term_ja)")

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


def _process_card_pipeline(request: AddCardRequest, output_dir: Path, index: int) -> dict:
    """Process a single card through the complete pipeline."""
    try:
        # Use the existing pipeline: infer missing fields -> process card
        complete_request = _infer_missing_fields(request)
        vocab_card = _process_single_card(complete_request, output_dir, index)
        return {'success': True, 'card': vocab_card}
    except Exception as e:
        return {
            'success': False,
            'error': f"Card '{request.term_en or request.term_ja}': {str(e)}"
        }


def _generate_audio_files(
    term_en: str, term_ja: str, output_dir: Path, index: int
) -> tuple[Path, Path]:
    """Generate TTS audio files for English and Japanese content.
    
    Args:
        term_en: English term for filename
        term_ja: Japanese term for filename  
        sentence_en: English sentence content for TTS
        sentence_ja: Japanese sentence content for TTS
        output_dir: Directory to save audio files
        index: Index for unique filenames
    
    Returns:
        Tuple of (english_audio_path, japanese_audio_path)
    """
    # Generate English audio
    audio_en_path = output_dir / f"{term_en.replace(' ', '_')}_{index}_en.mp3"
    en_result = generate_speech(content=term_en, language="English")
    with open(audio_en_path, "wb") as f:
        f.write(en_result.audio_bytes)

    # Generate Japanese audio
    audio_ja_path = output_dir / f"{term_ja.replace(' ', '_')}_{index}_ja.mp3"
    ja_result = generate_speech(content=term_ja, language="Japanese")
    with open(audio_ja_path, "wb") as f:
        f.write(ja_result.audio_bytes)

    return audio_en_path, audio_ja_path


def _process_single_card(request: AddCardRequest, output_dir: Path, index: int) -> AddVocabCardRequest:
    """Process a single card request into a complete AddVocabCardRequest."""
    # Infer missing fields using LLM
    complete_request = _infer_missing_fields(request)

    # Generate TTS audio files
    audio_en_path, audio_ja_path = _generate_audio_files(
        complete_request.term_en,
        complete_request.term_ja,
        output_dir=output_dir,
        index=index,
    )

    # Create vocab card request
    return AddVocabCardRequest(
        term_en=complete_request.term_en,
        term_ja=complete_request.term_ja,
        reading_ja=complete_request.reading_ja,
        sentence_en=complete_request.sentence_en,
        sentence_ja=complete_request.sentence_ja,
        audio_en=audio_en_path,
        audio_ja=audio_ja_path,
    )


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

    # Infer missing fields using LLM
    console.print("[yellow]Inferring missing fields...[/yellow]")
    complete_request = _infer_missing_fields(request)

    console.print("[green]Inferred fields:[/green]")
    console.print(f"  EN: {complete_request.term_en}")
    console.print(f"  JA: {complete_request.term_ja} ({complete_request.reading_ja})")
    console.print(f"  Example EN: {complete_request.sentence_en}")
    console.print(f"  Example JA: {complete_request.sentence_ja}")

    # Generate TTS audio
    audio_en_path = None
    audio_ja_path = None

    console.print("[yellow]Generating TTS audio...[/yellow]")

    # Generate English audio
    audio_en_path = output_dir / f"{complete_request.term_en.replace(' ', '_')}_en.mp3"
    en_result = generate_speech(
        content=complete_request.sentence_en, language="English"
    )

    # Write audio bytes to file
    with open(audio_en_path, "wb") as f:
        f.write(en_result.audio_bytes)

    # Generate Japanese audio
    audio_ja_path = output_dir / f"{complete_request.term_ja.replace(' ', '_')}_ja.mp3"
    ja_result = generate_speech(
        content=complete_request.sentence_ja, language="Japanese"
    )

    # Write audio bytes to file
    with open(audio_ja_path, "wb") as f:
        f.write(ja_result.audio_bytes)

    console.print(f"Generated audio files: EN: {audio_en_path}, JA: {audio_ja_path}")

    # Create the vocab card
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
    successful_results, failed_results = (from_iterable(enumerate(csv_rows, 1))
        .try_select(lambda item: _parse_csv_row(item[1], deck_name, item[0])))
    
    card_requests = list(successful_results)
    failed_cards = [str(exception) for exception in failed_results]

    console.print(f"[green]Found {len(card_requests)} valid cards to process[/green]")

    if failed_cards:
        console.print(f"[yellow]Skipped {len(failed_cards)} invalid rows[/yellow]")
        for failure in failed_cards:
            console.print(f"  [yellow]- {failure}[/yellow]")

    if len(card_requests) == 0:
        console.print("[yellow]No valid card requests to process[/yellow]")
        return

    # Create Table[AddCardRequest]
    card_requests_table = Table.from_pydantic(card_requests)

    # Always use the unified pipeline
    review_and_add(card_requests_table, deck_name, interactive=interactive)


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

    response = cast(
        ModelResponse,
        litellm.completion(
            model=ctx.config.fast_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "vocabulary_extraction",
                    "schema": VocabularyExtractionResponse.model_json_schema(),
                    "strict": True,
                },
            },
        ),
    )
    if not response.choices:
        raise RuntimeError("Failed to extract vocabulary from text")

    message_content = cast(litellm.Choices, response.choices)[0].message.content
    assert message_content is not None, "Response content is None"
    extraction_result = VocabularyExtractionResponse.model_validate_json(
        message_content
    )

    console.print(
        f"[green]Extracted {len(extraction_result.items)} vocabulary words[/green]"
    )

    # Convert vocabulary items to Table[AddCardRequest] using LINQ
    card_requests = (
        from_iterable(extraction_result.items)
        .select(
            lambda item: _vocab_item_to_card_request(
                item, source_language, target_language
            )
        )
        .to_list()
    )

    review_and_add(
        Table.from_pydantic(card_requests),
        deck_name,
        output_dir,
        interactive=interactive,
    )


@register()
def review_and_add(
    card_requests: Table,
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

    # Select which cards to process
    if interactive:
        # Use select_ui to let user choose which cards to create
        selected_requests = select_ui(
            card_requests,
            title="Card Requests - Select cards to create",
            display_columns=["term_en", "term_ja", "reading_ja"],
        )

        if len(selected_requests) == 0:
            console.print("[yellow]No cards selected[/yellow]")
            return

        console.print(
            f"[green]Selected {len(selected_requests)} cards to create[/green]"
        )
    else:
        # Non-interactive: process all cards
        selected_requests = card_requests

    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_review_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process selected requests using enhanced LINQ pipeline
    successful_results, failed_results = (from_iterable(enumerate(selected_requests))
        .with_progress("Processing selected cards")
        .try_select(lambda item: _process_card_pipeline(item[1], output_dir, item[0])))
    
    # Extract processed cards from successful results
    processed_cards = []
    for result in successful_results:
        if isinstance(result, dict) and result.get('success') and 'card' in result:
            processed_cards.append(result['card'])
    
    # Extract error messages from failed results
    failed_cards = [f"Processing error: {str(exception)}" for exception in failed_results]

    # Create batch deck with all processed cards
    console.print(f"[yellow]Creating deck with {len(processed_cards)} cards...[/yellow]")
    result = anki_add_vocab_cards(deck_name, processed_cards)
    cards_created = result.cards_created

    console.print(
        f"[green]Cards created: {cards_created}/{len(selected_requests)}[/green]"
    )

    if failed_cards:
        console.print(f"[red]Failed cards: {len(failed_cards)}[/red]")
        for failure in failed_cards:
            console.print(f"  [red]- {failure}[/red]")


if __name__ == "__main__":
    functions = [add_card, add_from_csv, add_from_text, review_and_add]
    cli_main(
        functions,
        context_cls=ToolContext,
    )

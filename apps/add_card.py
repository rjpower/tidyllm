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
from rich.progress import track
from rich.table import Table

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.anki import (
    AddVocabCardRequest,
    AddVocabCardsRequest,
    AnkiCreateResult,
    anki_add_vocab_cards,
)
from tidyllm.tools.context import ToolContext
from tidyllm.tools.tts import generate_speech

console = Console()


class AddCardRequest(BaseModel):
    """Request to add a single vocabulary card."""

    term_en: str = ""
    term_ja: str = ""
    reading_ja: str = ""
    sentence_en: str = ""
    sentence_ja: str = ""
    deck_name: str = "Japanese Vocabulary"
    output_dir: Path | None = None


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
            response_format={"type": "json_object"},
        ),
    )

    message_content = cast(litellm.Choices, response.choices)[0].message.content
    assert message_content is not None, "Response content is None"
    completed_data = json.loads(message_content)

    # Merge with original request, preserving non-empty original values
    final_data = request_data.copy()
    for key, value in completed_data.items():
        if key in final_data and (not final_data[key] or final_data[key] == ""):
            final_data[key] = value

    # Re-add the output_dir from the original request
    final_data["output_dir"] = request.output_dir
    final_data["deck_name"] = request.deck_name

    return AddCardRequest(**final_data)


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


class VocabularyExtractionResponse(BaseModel):
    """Response model for vocabulary extraction from text."""
    items: list[VocabularyItem]


def _generate_audio_files(
    term_en: str, term_ja: str, sentence_en: str, sentence_ja: str, 
    output_dir: Path, index: int
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
    en_result = generate_speech(content=sentence_en, language="English")
    with open(audio_en_path, "wb") as f:
        f.write(en_result.audio_bytes)

    # Generate Japanese audio
    audio_ja_path = output_dir / f"{term_ja.replace(' ', '_')}_{index}_ja.mp3"
    ja_result = generate_speech(content=sentence_ja, language="Japanese")
    with open(audio_ja_path, "wb") as f:
        f.write(ja_result.audio_bytes)

    return audio_en_path, audio_ja_path


def _process_single_card(request: AddCardRequest, output_dir: Path, index: int) -> AddVocabCardRequest:
    """Process a single card request into a complete AddVocabCardRequest.
    
    Args:
        request: Basic card request with potentially missing fields
        output_dir: Directory for generated audio files
        index: Index for unique filenames
    
    Returns:
        Complete AddVocabCardRequest ready for deck creation
    """
    # Infer missing fields using LLM
    complete_request = _infer_missing_fields(request)
    
    # Generate TTS audio files
    audio_en_path, audio_ja_path = _generate_audio_files(
        complete_request.term_en,
        complete_request.term_ja, 
        complete_request.sentence_en,
        complete_request.sentence_ja,
        output_dir,
        index
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


def _create_batch_deck(cards: list[AddVocabCardRequest], deck_name: str) -> AnkiCreateResult:
    """Create an Anki deck with multiple cards.
    
    Args:
        cards: List of vocabulary cards to add
        deck_name: Name of the deck
    
    Returns:
        AnkiCreateResult with creation details
    """
    if not cards:
        return AnkiCreateResult(
            deck_path=Path(tempfile.gettempdir()) / f"{deck_name.replace(' ', '_')}_vocab.apkg",
            cards_created=0,
            message="No cards to create"
        )
    
    vocab_request = AddVocabCardsRequest(cards=cards, deck_name=deck_name)
    return anki_add_vocab_cards(vocab_request)


@register()
def add_card(
    term_en: str = "",
    term_ja: str = "",
    deck_name: str = "Japanese Vocabulary",
) -> AddCardResult:
    """Add a vocabulary card with auto-generated content.

    All fields except the source term are automatically inferred using LLM.

    Args:
        term_en: English term (either term_en or term_ja must be provided)
        term_ja: Japanese term (either term_en or term_ja must be provided)
        deck_name: Name of the Anki deck

    Returns:
        AddCardResult with creation details

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
        deck_name=deck_name,
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

    console.print("[green]Generated audio files:[/green]")
    console.print(f"  EN: {audio_en_path}")
    console.print(f"  JA: {audio_ja_path}")

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

    vocab_request = AddVocabCardsRequest(
        cards=[vocab_card],
        deck_name=complete_request.deck_name,
    )

    result = anki_add_vocab_cards(vocab_request)

    console.print("[green]Card created successfully![/green]")
    console.print(f"[green]Deck file:[/green] {result.deck_path}")

    return AddCardResult(
        card_created=True,
        deck_path=result.deck_path,
        message=result.message
    )


@register()
def add_from_csv(
    csv_path: Path,
    deck_name: str = "Japanese Vocabulary",
) -> BatchAddResult:
    """Add vocabulary cards from CSV file.

    Expected CSV format (only one term per row required, others are optional):
    term_en,term_ja,reading_ja,sentence_en,sentence_ja
    hello,こんにちは,こんにちは,,
    ,ありがとう,ありがとう,,

    Missing fields will be automatically inferred by the LLM.

    Args:
        csv_path: Path to CSV file
        deck_name: Name of the Anki deck

    Returns:
        BatchAddResult with creation statistics

    Example: add_from_csv(Path("vocab.csv"), "Japanese::N5")
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    console.print(f"[bold blue]Processing CSV file:[/bold blue] {csv_path}")

    # Setup output directory
    output_dir = Path(tempfile.mkdtemp(prefix="add_card_batch_"))

    # Read CSV file
    cards_data = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards_data.append({
                'term_en': row.get('term_en', '').strip(),
                'term_ja': row.get('term_ja', '').strip(),
                'reading_ja': row.get('reading_ja', '').strip(),
                'sentence_en': row.get('sentence_en', '').strip(),
                'sentence_ja': row.get('sentence_ja', '').strip(),
            })

    console.print(f"[green]Found {len(cards_data)} cards to process[/green]")

    # Process each card and collect them for batch creation
    processed_cards = []
    failed_cards = []

    for i, card_data in enumerate(track(cards_data, description="Processing cards")):
        try:
            # At least one term (English or Japanese) must be provided
            if not card_data['term_en'] and not card_data['term_ja']:
                failed_cards.append(f"Row {i+1}: Missing required term (either term_en or term_ja)")
                continue

            # Create request with provided terms
            request = AddCardRequest(
                term_en=card_data["term_en"],
                term_ja=card_data["term_ja"],
                deck_name=deck_name,
            )

            # Process the card using helper function
            vocab_card = _process_single_card(request, output_dir, i)
            processed_cards.append(vocab_card)

        except Exception as e:
            failed_cards.append(f"Row {i+1} ({card_data.get('term_en', card_data.get('term_ja', 'unknown'))}): {str(e)}")

    # Create batch deck with all processed cards
    console.print(f"[yellow]Creating deck with {len(processed_cards)} cards...[/yellow]")
    result = _create_batch_deck(processed_cards, deck_name)
    deck_path = result.deck_path
    cards_created = result.cards_created

    console.print("\n[bold green]Batch processing complete![/bold green]")
    console.print(f"[green]Cards created: {cards_created}/{len(cards_data)}[/green]")

    if failed_cards:
        console.print(f"[red]Failed cards: {len(failed_cards)}[/red]")
        for failure in failed_cards:
            console.print(f"  [red]- {failure}[/red]")

    return BatchAddResult(
        cards_created=cards_created,
        total_cards=len(cards_data),
        deck_path=deck_path,
        failed_cards=failed_cards,
    )


@register()
def add_from_text(
    text: str,
    source_language: str = "ja",
    target_language: str = "en",
    deck_name: str = "Japanese Vocabulary",
    max_words: int = 10,
    output_dir: Path | None = None,
) -> BatchAddResult:
    """Extract vocabulary from text and create cards.

    Args:
        text: Text to extract vocabulary from
        source_language: Source language of the text
        target_language: Target language for translations
        deck_name: Name of the Anki deck
        max_words: Maximum number of words to extract
        output_dir: Output directory for files

    Returns:
        BatchAddResult with creation statistics

    Example: add_from_text("日本語の文章です", "ja", "en", "Japanese::Reading")
    """
    console.print("[bold blue]Extracting vocabulary from text[/bold blue]")
    console.print(f"[blue]Text:[/blue] {text[:100]}{'...' if len(text) > 100 else ''}")

    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_text_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Use LLM to extract vocabulary
    import litellm

    from tidyllm.context import get_tool_context

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
    vocab_data = [item.model_dump() for item in extraction_result.items]

    console.print(f"[green]Extracted {len(vocab_data)} vocabulary words[/green]")

    # Show vocabulary table
    table = Table(title="Extracted Vocabulary")
    table.add_column("Word", style="cyan")
    table.add_column("Translation", style="magenta")
    table.add_column("Reading", style="green")

    for item in vocab_data:
        table.add_row(
            item["word"],
            item["translation"],
            item["reading"]
        )

    console.print(table)

    # Process each vocabulary item and collect them for batch creation
    processed_cards = []
    failed_cards = []

    for i, item in enumerate(track(vocab_data, description="Processing vocabulary")):
        try:
            if source_language == "ja" and target_language == "en":
                term_ja = item["word"]
                term_en = item["translation"]
                reading_ja = item["reading"]
            else:
                term_en = item["word"]
                term_ja = item["translation"]
                reading_ja = item["reading"]

            # Create request for processing
            request = AddCardRequest(
                term_en=term_en,
                term_ja=term_ja,
                reading_ja=reading_ja,
                deck_name=deck_name,
            )

            # Process the card using helper function
            vocab_card = _process_single_card(request, output_dir, i)
            processed_cards.append(vocab_card)

        except Exception as e:
            failed_cards.append(f"Word '{item['word']}': {str(e)}")

    # Create batch deck with all processed cards
    console.print(f"[yellow]Creating deck with {len(processed_cards)} cards...[/yellow]")
    result = _create_batch_deck(processed_cards, deck_name)
    deck_path = result.deck_path
    cards_created = result.cards_created

    console.print("\n[bold green]Text processing complete![/bold green]")
    console.print(f"[green]Cards created: {cards_created}/{len(vocab_data)}[/green]")

    if failed_cards:
        console.print(f"[red]Failed cards: {len(failed_cards)}[/red]")
        for failure in failed_cards:
            console.print(f"  [red]- {failure}[/red]")

    return BatchAddResult(
        cards_created=cards_created,
        total_cards=len(vocab_data),
        deck_path=deck_path,
        failed_cards=failed_cards,
    )


@register()
def review_and_add(
    terms: list[str],
    deck_name: str = "Japanese Vocabulary",
    output_dir: Path | None = None,
) -> BatchAddResult:
    """Interactive review and selection of terms before adding cards.

    Args:
        terms: List of terms to review (format: "english:japanese" or "japanese")
        deck_name: Name of the Anki deck
        output_dir: Output directory for files

    Returns:
        BatchAddResult with creation statistics

    Example: review_and_add(["hello:こんにちは", "thank you:ありがとう"])
    """
    console.print(f"[bold blue]Interactive review of {len(terms)} terms[/bold blue]")

    # Parse terms
    parsed_terms = []
    for term in terms:
        if ':' in term:
            en, ja = term.split(':', 1)
            parsed_terms.append({'term_en': en.strip(), 'term_ja': ja.strip()})
        else:
            # Auto-detect and translate
            parsed_terms.append({'term_en': '', 'term_ja': term.strip()})

    # Show terms table
    table = Table(title="Terms to Review")
    table.add_column("Index", style="cyan")
    table.add_column("English", style="magenta")
    table.add_column("Japanese", style="green")

    for i, term in enumerate(parsed_terms):
        table.add_row(
            str(i),
            term['term_en'] or '[italic]auto-detect[/italic]',
            term['term_ja']
        )

    console.print(table)

    # Get user selection
    console.print("\n[yellow]Enter term indices to add (comma-separated), or 'all' for all terms:[/yellow]")
    selection = console.input("Selection [all]: ") or "all"

    if selection.lower() == "all":
        selected_terms = parsed_terms
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
            selected_terms = [parsed_terms[i] for i in indices if 0 <= i < len(parsed_terms)]
        except ValueError:
            console.print("[red]Invalid selection format[/red]")
            return BatchAddResult(cards_created=0, total_cards=0, deck_path=Path(), failed_cards=["Invalid selection"])

    console.print(f"[green]Selected {len(selected_terms)} terms to add[/green]")

    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_review_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process selected terms for batch creation
    processed_cards = []
    failed_cards = []

    for i, term in enumerate(track(selected_terms, description="Processing cards")):
        try:
            # Auto-detect missing translations if needed
            if not term['term_en']:
                # Use LLM to translate
                import litellm

                from tidyllm.context import get_tool_context

                ctx = get_tool_context()
                response = cast(
                    ModelResponse,
                    litellm.completion(
                        model=ctx.config.fast_model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Translate '{term['term_ja']}' to English. Return only the translation.",
                            }
                        ],
                    ),
                )
                if response.choices:
                    message_content = cast(litellm.Choices, response.choices)[0].message.content
                    assert message_content is not None, "Response content is None"
                    term["term_en"] = message_content.strip()
                else:
                    term['term_en'] = "translation_needed"

            # Create request for processing
            request = AddCardRequest(
                term_en=term["term_en"],
                term_ja=term["term_ja"],
                deck_name=deck_name,
            )

            # Process the card using helper function
            vocab_card = _process_single_card(request, output_dir, i)
            processed_cards.append(vocab_card)

        except Exception as e:
            failed_cards.append(f"Term '{term['term_ja']}': {str(e)}")

    # Create batch deck with all processed cards
    console.print(f"[yellow]Creating deck with {len(processed_cards)} cards...[/yellow]")
    result = _create_batch_deck(processed_cards, deck_name)
    deck_path = result.deck_path
    cards_created = result.cards_created

    console.print("\n[bold green]Review complete![/bold green]")
    console.print(f"[green]Cards created: {cards_created}/{len(selected_terms)}[/green]")

    if failed_cards:
        console.print(f"[red]Failed cards: {len(failed_cards)}[/red]")
        for failure in failed_cards:
            console.print(f"  [red]- {failure}[/red]")

    return BatchAddResult(
        cards_created=cards_created,
        total_cards=len(selected_terms),
        deck_path=deck_path,
        failed_cards=failed_cards,
    )


if __name__ == "__main__":
    functions = [add_card, add_from_csv, add_from_text, review_and_add]
    cli_main(
        functions,
        context_cls=ToolContext,
    )

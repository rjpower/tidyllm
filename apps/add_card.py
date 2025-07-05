#!/usr/bin/env python3
"""Anki flashcard creation app with TTS audio generation.

This app creates bilingual vocabulary flashcards with generated example sentences
and TTS audio for both languages, then creates Anki packages for import.
"""

import csv
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from rich.table import Table

from tidyllm.adapters.cli import cli_main
from tidyllm.registry import register
from tidyllm.tools.anki import (
    AddVocabCardRequest,
    AnkiCreateResult,
    anki_add_vocab_card,
    generate_example_sentence,
)
from tidyllm.tools.context import ToolContext
from tidyllm.tools.tts import generate_speech_file

console = Console()


def _infer_missing_fields(request: AddCardRequest) -> AddCardRequest:
    """Infer missing fields for a vocabulary card using LLM.

    Args:
        request: AddCardRequest with potentially missing fields

    Returns:
        AddCardRequest with all fields completed
    """
    import json

    import litellm

    from tidyllm.context import get_tool_context

    ctx = get_tool_context()

    # Convert request to dict for processing
    request_data = request.model_dump(exclude_unset=False)

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

    response = litellm.completion(
        model=ctx.config.fast_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    if not response.choices:
        raise RuntimeError("Failed to infer missing fields")

    try:
        completed_data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON response: {str(e)}")

    # Merge with original request, preserving non-empty original values
    final_data = request_data.copy()
    for key, value in completed_data.items():
        if key in final_data and (not final_data[key] or final_data[key] == ""):
            final_data[key] = value

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


class AddCardRequest(BaseModel):
    """Request to add a single vocabulary card."""
    term_en: str = ""
    term_ja: str = ""
    reading_ja: str = ""
    sentence_en: str = ""
    sentence_ja: str = ""
    deck_name: str = "Japanese Vocabulary"
    generate_audio: bool = True
    output_dir: Path | None = None


@register()
def add_card(
    term: str,
    deck_name: str = "Japanese Vocabulary",
    generate_audio: bool = True,
    output_dir: Path | None = None,
) -> AddCardResult:
    """Add a vocabulary card with auto-generated content.
    
    All fields except the source term are automatically inferred using LLM.
    
    Args:
        term: Either English or Japanese term (required)
        deck_name: Name of the Anki deck
        generate_audio: Whether to generate TTS audio
        output_dir: Output directory for files
        
    Returns:
        AddCardResult with creation details
        
    Example: add_card("こんにちは")
    Example: add_card("hello", deck_name="English::Basic")
    """
    console.print(f"[bold blue]Creating card for:[/bold blue] {term}")
    
    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if term is English or Japanese and create request
    import re
    is_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', term))
    
    if is_japanese:
        request = AddCardRequest(
            term_ja=term,
            deck_name=deck_name,
            generate_audio=generate_audio,
            output_dir=output_dir
        )
    else:
        request = AddCardRequest(
            term_en=term,
            deck_name=deck_name,
            generate_audio=generate_audio,
            output_dir=output_dir
        )
    
    # Infer missing fields using LLM
    console.print("[yellow]Inferring missing fields...[/yellow]")
    complete_request = _infer_missing_fields(request)
    
    console.print(f"[green]Inferred fields:[/green]")
    console.print(f"  EN: {complete_request.term_en}")
    console.print(f"  JA: {complete_request.term_ja} ({complete_request.reading_ja})")
    console.print(f"  Example EN: {complete_request.sentence_en}")
    console.print(f"  Example JA: {complete_request.sentence_ja}")
    
    # Generate TTS audio if requested
    audio_en_path = None
    audio_ja_path = None
    
    if complete_request.generate_audio:
        console.print("[yellow]Generating TTS audio...[/yellow]")
        
        # Generate English audio
        audio_en_path = output_dir / f"{complete_request.term_en.replace(' ', '_')}_en.mp3"
        generate_speech_file(
            content=complete_request.sentence_en,
            output_path=audio_en_path,
            auto_detect_language=True
        )
        
        # Generate Japanese audio
        audio_ja_path = output_dir / f"{complete_request.term_ja.replace(' ', '_')}_ja.mp3"
        generate_speech_file(
            content=complete_request.sentence_ja,
            output_path=audio_ja_path,
            auto_detect_language=True
        )
        
        console.print(f"[green]Generated audio files:[/green]")
        console.print(f"  EN: {audio_en_path}")
        console.print(f"  JA: {audio_ja_path}")
    
    # Create the vocab card
    console.print("[yellow]Creating Anki card...[/yellow]")
    
    vocab_request = AddVocabCardRequest(
        term_en=complete_request.term_en,
        term_ja=complete_request.term_ja,
        reading_ja=complete_request.reading_ja,
        sentence_en=complete_request.sentence_en,
        sentence_ja=complete_request.sentence_ja,
        audio_en=audio_en_path,
        audio_ja=audio_ja_path,
        deck_name=complete_request.deck_name,
    )
    
    result = anki_add_vocab_card(vocab_request)
    
    console.print(f"[green]Card created successfully![/green]")
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
    generate_audio: bool = True,
    output_dir: Path | None = None,
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
        generate_audio: Whether to generate TTS audio
        output_dir: Output directory for files
        
    Returns:
        BatchAddResult with creation statistics
        
    Example: add_from_csv(Path("vocab.csv"), "Japanese::N5", generate_audio=True)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    console.print(f"[bold blue]Processing CSV file:[/bold blue] {csv_path}")
    
    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_batch_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    cards_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
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
    
    # Process each card
    cards_created = 0
    failed_cards = []
    
    for i, card_data in enumerate(track(cards_data, description="Creating cards")):
        try:
            # At least one term (English or Japanese) must be provided
            if not card_data['term_en'] and not card_data['term_ja']:
                failed_cards.append(f"Row {i+1}: Missing required term (either term_en or term_ja)")
                continue
            
            # Create individual card using the main term
            term = card_data['term_en'] or card_data['term_ja']
            card_output_dir = output_dir / f"card_{i+1}"
            add_card(
                term=term,
                deck_name=deck_name,
                generate_audio=generate_audio,
                output_dir=card_output_dir,
            )
            cards_created += 1
            
        except Exception as e:
            failed_cards.append(f"Row {i+1} ({card_data['term_en']}): {str(e)}")
    
    # Find the final deck path (should be the same for all cards)
    deck_path = output_dir / f"{deck_name.replace(' ', '_')}_vocab.apkg"
    
    console.print(f"\n[bold green]Batch processing complete![/bold green]")
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
    generate_audio: bool = True,
    max_words: int = 10,
    output_dir: Path | None = None,
) -> BatchAddResult:
    """Extract vocabulary from text and create cards.
    
    Args:
        text: Text to extract vocabulary from
        source_language: Source language of the text
        target_language: Target language for translations
        deck_name: Name of the Anki deck
        generate_audio: Whether to generate TTS audio
        max_words: Maximum number of words to extract
        output_dir: Output directory for files
        
    Returns:
        BatchAddResult with creation statistics
        
    Example: add_from_text("日本語の文章です", "ja", "en", "Japanese::Reading")
    """
    console.print(f"[bold blue]Extracting vocabulary from text[/bold blue]")
    console.print(f"[blue]Text:[/blue] {text[:100]}{'...' if len(text) > 100 else ''}")

    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="add_card_text_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Use LLM to extract vocabulary
    import json

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

Return a JSON array of vocabulary items:
[
    {{
        "word": "word in source language",
        "translation": "word in target language", 
        "reading": "pronunciation/reading if applicable"
    }}
]"""

    response = litellm.completion(
        model=ctx.config.fast_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "vocabulary_extraction",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "translation": {"type": "string"},
                            "reading": {"type": "string"}
                        },
                        "required": ["word", "translation", "reading"]
                    }
                },
                "strict": True
            }
        }
    )

    if not response.choices:
        raise RuntimeError("Failed to extract vocabulary from text")

    vocab_data = json.loads(response.choices[0].message.content)

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

    # Create cards for each vocabulary item
    cards_created = 0
    failed_cards = []

    for i, item in enumerate(track(vocab_data, description="Creating cards")):
        try:
            if source_language == "ja" and target_language == "en":
                term_ja = item["word"]
                term_en = item["translation"]
                reading_ja = item["reading"]
            else:
                term_en = item["word"]
                term_ja = item["translation"]
                reading_ja = item["reading"]

            # Create individual card
            card_output_dir = output_dir / f"card_{i+1}"
            term = term_ja if source_language == "ja" else term_en
            add_card(
                term=term,
                deck_name=deck_name,
                generate_audio=generate_audio,
                output_dir=card_output_dir,
            )
            cards_created += 1

        except Exception as e:
            failed_cards.append(f"Word '{item['word']}': {str(e)}")

    # Find the final deck path
    deck_path = output_dir / f"{deck_name.replace(' ', '_')}_vocab.apkg"

    console.print(f"\n[bold green]Text processing complete![/bold green]")
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
    generate_audio: bool = True,
    output_dir: Path | None = None,
) -> BatchAddResult:
    """Interactive review and selection of terms before adding cards.
    
    Args:
        terms: List of terms to review (format: "english:japanese" or "japanese")
        deck_name: Name of the Anki deck
        generate_audio: Whether to generate TTS audio
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

    # Create cards for selected terms
    cards_created = 0
    failed_cards = []

    for i, term in enumerate(track(selected_terms, description="Creating cards")):
        try:
            # Auto-detect missing translations if needed
            if not term['term_en']:
                # Use LLM to translate
                import litellm

                from tidyllm.context import get_tool_context

                ctx = get_tool_context()
                response = litellm.completion(
                    model=ctx.config.fast_model,
                    messages=[{"role": "user", "content": f"Translate '{term['term_ja']}' to English. Return only the translation."}]
                )

                if response.choices:
                    term['term_en'] = response.choices[0].message.content.strip()
                else:
                    term['term_en'] = "translation_needed"

            # Create individual card
            card_output_dir = output_dir / f"card_{i+1}"
            main_term = term['term_ja'] or term['term_en']
            add_card(
                term=main_term,
                deck_name=deck_name,
                generate_audio=generate_audio,
                output_dir=card_output_dir,
            )
            cards_created += 1

        except Exception as e:
            failed_cards.append(f"Term '{term['term_ja']}': {str(e)}")

    # Find the final deck path
    deck_path = output_dir / f"{deck_name.replace(' ', '_')}_vocab.apkg"

    console.print(f"\n[bold green]Review complete![/bold green]")
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

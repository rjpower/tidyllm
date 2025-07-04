#!/usr/bin/env python3
"""Long-form audio transcription and vocabulary extraction tools.

This module provides tools for processing audio files through VAD segmentation,
transcription, vocabulary extraction, and interactive review for adding new
words to the vocabulary database.
"""

import csv
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from rich.table import Table

from tidyllm.adapters.cli import multi_cli_main
from tidyllm.registry import register
from tidyllm.tools.audio import chunk_by_vad_stream, chunk_to_wav_bytes
from tidyllm.tools.audio import file as audio_file
from tidyllm.tools.context import ToolContext
from tidyllm.tools.transcribe import (
    TranscribedWord,
    TranscriptionResult,
    transcribe_bytes,
)
from tidyllm.tools.vocab_table import vocab_add, vocab_search

console = Console()


class SegmentTranscription(BaseModel):
    """Transcription result for a single audio segment."""

    segment_index: int
    start_time: float
    result: TranscriptionResult


class TranscribeAudioResult(BaseModel):
    """Result of audio transcription."""

    transcriptions: list[SegmentTranscription]
    total_segments: int
    total_words: int
    source_language: str
    target_language: str


class DiffVocabResult(BaseModel):
    """Result of vocabulary diffing."""

    new_words: list[dict]
    existing_count: int
    new_count: int


class ReviewVocabResult(BaseModel):
    """Result of vocabulary review."""

    added_count: int
    total_words: int


class FullPipelineResult(BaseModel):
    """Result of full pipeline."""

    total_segments: int
    total_words: int
    new_words: int
    added_words: int


@register()
def transcribe_audio(
    audio_path: Path,
    source_language: str | None = None,
    target_language: str = "en",
    output: Path | None = None,
) -> TranscribeAudioResult:
    """Transcribe audio file with VAD segmentation and extract vocabulary.

    Args:
        audio_path: Path to audio file
        source_language: Source language (auto-detect if not provided)
        target_language: Target language for translation
        output: File to write output to. If not specified, writes to stdout

    Returns:
        TranscribeAudioResult containing transcriptions and metadata

    Example: transcribe_audio(Path("speech.mp3"), target_language="en")
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    console.print(f"[bold blue]Transcribing audio:[/bold blue] {audio_path}")

    # Step 1: Segment audio using VAD
    console.print("[yellow]Segmenting audio using Voice Activity Detection...[/yellow]")
    audio_stream = audio_file(audio_path)
    segments = chunk_by_vad_stream(audio_stream)

    # Step 2: Transcribe each segment
    all_transcriptions = []
    all_words = []

    for segment in track(segments, description="Transcribing segments"):
        # Convert AudioChunk to WAV bytes (no temporary files!)
        wav_bytes = chunk_to_wav_bytes(segment)

        # Transcribe segment using bytes-based function (cacheable!)
        transcription_result = transcribe_bytes(
            wav_bytes,
            "audio/wav",
            source_language=source_language,
            target_language=target_language,
        )
        print(
            "Transcribed: ", segment.timestamp, segment.duration, transcription_result
        )

        segment_transcription = SegmentTranscription(
            segment_index=len(all_transcriptions),
            start_time=segment.timestamp,
            result=transcription_result,
        )

        all_transcriptions.append(segment_transcription)
        all_words.extend(transcription_result.words)

    console.print(f"[green]Transcribed {len(all_transcriptions)} segments[/green]")
    console.print(f"[green]Extracted {len(all_words)} vocabulary words[/green]")

    final_result = TranscribeAudioResult(
        transcriptions=all_transcriptions,
        total_segments=len(all_transcriptions),
        total_words=len(all_words),
        source_language=source_language or "auto-detected",
        target_language=target_language,
    )

    if output:
        output.write_text(final_result.model_dump_json())
        console.print(f"[green]Transcriptions saved to:[/green] {output}")
    else:
        console.print(final_result.model_dump_json(indent=2))

    return final_result

@register()
def diff_vocab(
    transcription_file: Path,
    output: Path | None = None,
) -> DiffVocabResult:
    """Find new vocabulary words not in database.

    Args:
        transcription_file: Path to transcription JSON file
        output: Output file for new words

    Returns:
        DiffVocabResult containing new words and counts

    Example: diff_vocab(Path("transcriptions.json"))
    """
    if not transcription_file.exists():
        raise FileNotFoundError(f"Transcription file not found: {transcription_file}")

    console.print("[bold blue]Comparing vocabulary against database...[/bold blue]")

    # Load transcription data
    with open(transcription_file) as f:
        data = json.load(f)

    # Extract all words
    all_words = []
    for transcription in data["transcriptions"]:
        for word in transcription["result"]["words"]:
            if word["word_native"] and word["word_translated"]:
                all_words.append(
                    TranscribedWord(
                        word_native=word["word_native"],
                        word_translated=word["word_translated"],
                    )
                )

    # Check against existing vocabulary
    new_words = []
    existing_count = 0

    for word in track(all_words, description="Checking vocabulary"):
        search_result = vocab_search({"word": word.word_native, "limit": 1})
        if not search_result.items:
            new_words.append(
                {"word": word.word_native, "translation": word.word_translated}
            )
        else:
            existing_count += 1

    console.print(f"[green]Found {len(new_words)} new words[/green]")
    console.print(f"[yellow]{existing_count} words already in database[/yellow]")

    result = DiffVocabResult(
        new_words=new_words, existing_count=existing_count, new_count=len(new_words)
    )

    if output:
        with open(output, "w") as f:
            json.dump(new_words, f, indent=2, ensure_ascii=False)
        console.print(f"[green]New words saved to:[/green] {output}")

    return result


@register()
def review_vocab(
    new_words_file: Path,
    auto_add: bool = False,
) -> ReviewVocabResult:
    """Interactive review and selection of vocabulary to add.

    Args:
        new_words_file: Path to new words JSON file
        auto_add: Automatically add all words without review

    Returns:
        ReviewVocabResult containing counts

    Example: review_vocab(Path("new_words.json"), auto_add=False)
    """
    if not new_words_file.exists():
        raise FileNotFoundError(f"New words file not found: {new_words_file}")

    console.print("[bold blue]Reviewing vocabulary for addition...[/bold blue]")

    # Load new words
    with open(new_words_file) as f:
        words_data = json.load(f)

    if not words_data:
        console.print("[yellow]No new words to review[/yellow]")
        return ReviewVocabResult(added_count=0, total_words=0)

    if auto_add:
        # Add all words automatically
        added_count = 0
        for word_data in track(words_data, description="Adding words"):
            try:
                vocab_add({
                    "word": word_data["word"],
                    "translation": word_data["translation"],
                    "tags": ["transcribed"]
                })
                added_count += 1
            except Exception:
                pass  # Skip failed additions

        console.print(f"[green]Added {added_count} words to vocabulary[/green]")
        return ReviewVocabResult(
            added_count=added_count, total_words=len(words_data)
        )

    # Interactive review
    table = Table(title="New Vocabulary Words")
    table.add_column("Index", style="cyan")
    table.add_column("Word", style="magenta")
    table.add_column("Translation", style="green")

    for i, word_data in enumerate(words_data):
        table.add_row(str(i), word_data["word"], word_data["translation"])

    console.print(table)

    # Get user selection
    console.print("\n[yellow]Enter word indices to add (comma-separated), or 'all' for all words:[/yellow]")
    selection = console.input("Selection [all]: ") or "all"

    if selection.lower() == "all":
        indices = list(range(len(words_data)))
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
        except ValueError:
            console.print("[red]Invalid selection format[/red]")
            raise ValueError("Invalid selection format")

    # Add selected words
    added_count = 0
    for i in indices:
        if 0 <= i < len(words_data):
            word_data = words_data[i]
            try:
                vocab_add({
                    "word": word_data["word"],
                    "translation": word_data["translation"],
                    "tags": ["transcribed"]
                })
                added_count += 1
                console.print(f"[green]Added:[/green] {word_data['word']} -> {word_data['translation']}")
            except Exception:
                console.print(f"[red]Failed to add:[/red] {word_data['word']}")

    console.print(f"\n[green]Added {added_count} words to vocabulary[/green]")
    return ReviewVocabResult(
        added_count=added_count, total_words=len(words_data)
    )


class ExportCsvResult(BaseModel):
    """Result of CSV export."""

    exported_count: int
    output_file: str


@register()
def export_csv(
    new_words_file: Path,
    output_csv: Path,
) -> ExportCsvResult:
    """Export new vocabulary words to CSV.

    Args:
        new_words_file: Path to new words JSON file
        output_csv: Output CSV file path

    Returns:
        ExportCsvResult containing export counts and success status

    Example: export_csv(Path("new_words.json"), Path("vocabulary.csv"))
    """
    if not new_words_file.exists():
        raise FileNotFoundError(f"New words file not found: {new_words_file}")

    console.print(f"[bold blue]Exporting to CSV:[/bold blue] {output_csv}")

    # Load new words
    with open(new_words_file) as f:
        words_data = json.load(f)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ['word', 'translation', 'source_language', 'target_language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for word_data in words_data:
            writer.writerow({
                'word': word_data['word'],
                'translation': word_data['translation'],
                'source_language': 'auto-detected',
                'target_language': 'en'
            })

    console.print(f"[green]Exported {len(words_data)} words to CSV[/green]")
    return ExportCsvResult(
        exported_count=len(words_data), output_file=str(output_csv)
    )

@register()
def full_pipeline(
    audio_path: Path,
    source_language: str | None = None,
    target_language: str = "en",
    output_dir: Path | None = None,
    auto_add: bool = False,
) -> FullPipelineResult:
    """Run complete transcription and vocabulary extraction pipeline.

    Args:
        audio_path: Path to audio file
        source_language: Source language
        target_language: Target language
        output_dir: Output directory for intermediate files
        auto_add: Automatically add all new words

    Returns:
        FullPipelineResult containing pipeline statistics

    Example: full_pipeline(Path("speech.mp3"), auto_add=True)
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    console.print(f"[bold blue]Running full pipeline on:[/bold blue] {audio_path}")

    # Setup output directory
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    else:
        output_dir.mkdir(exist_ok=True)

    transcription_file = output_dir / "transcriptions.json"
    new_words_file = output_dir / "new_words.json"
    csv_file = output_dir / "new_words.csv"

    console.print(f"[yellow]Working directory:[/yellow] {output_dir}")

    # Step 1: Transcribe audio with segmentation
    console.print("\n[bold]Step 1: Transcribing audio with VAD segmentation[/bold]")
    transcribe_result = transcribe_audio(
        audio_path=audio_path,
        source_language=source_language,
        target_language=target_language,
        output=transcription_file,
    )

    # Step 2: Find new vocabulary
    console.print("\n[bold]Step 2: Finding new vocabulary[/bold]")
    diff_result = diff_vocab(
        transcription_file=transcription_file,
        output=new_words_file,
    )

    # Step 3: Export to CSV
    console.print("\n[bold]Step 3: Exporting to CSV[/bold]")
    export_csv(
        new_words_file=new_words_file,
        output_csv=csv_file,
    )

    # Step 4: Review and add vocabulary
    console.print("\n[bold]Step 4: Adding vocabulary to database[/bold]")
    review_result = review_vocab(
        new_words_file=new_words_file,
        auto_add=auto_add,
    )

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"[green]Results saved in:[/green] {output_dir}")

    return FullPipelineResult(
        total_segments=transcribe_result.total_segments,
        total_words=transcribe_result.total_words,
        new_words=diff_result.new_count,
        added_words=review_result.added_count,
    )


if __name__ == "__main__":
    functions = [transcribe_audio, diff_vocab, review_vocab, export_csv, full_pipeline]
    multi_cli_main(
        functions,
        default_function="transcribe_audio",
        context_cls=ToolContext,
    )

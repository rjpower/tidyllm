#!/usr/bin/env python3
"""Long-form audio transcription and vocabulary extraction tools.

This module provides tools for processing audio files through VAD segmentation,
transcription, vocabulary extraction, and interactive review for adding new
words to the vocabulary database.
"""

import csv
import json
import tempfile
import traceback
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console

from tidyllm.adapters.cli import cli_main
from tidyllm.duration import Duration
from tidyllm.linq import Enumerable, Table, from_iterable
from tidyllm.registry import register
from tidyllm.serialization import to_json_dict
from tidyllm.source import SourceLike
from tidyllm.tools.audio import (
    audio_from_source,
    chunk_by_vad_stream,
    chunk_to_wav_bytes,
)
from tidyllm.tools.context import ToolContext
from tidyllm.tools.transcribe import (
    TranscribedWord,
    TranscriptionResult,
    transcribe_audio,
)
from tidyllm.tools.vocab_table import vocab_add, vocab_search
from tidyllm.ui.selection import select_ui

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


class FullPipelineResult(BaseModel):
    """Result of full pipeline."""

    total_segments: int
    total_words: int
    new_words: int


@register()
def transcribe_with_vad(
    audio_source: SourceLike,
    source_language: str | None = None,
    target_language: str = "en",
    output: Path | None = None,
) -> TranscribeAudioResult:
    """Transcribe audio from any source with VAD segmentation and extract vocabulary.

    Args:
        audio_source: Audio source (file path, bytes, URL, etc.)
        source_language: Source language (auto-detect if not provided)
        target_language: Target language for translation
        output: File to write output to. If not specified, writes to stdout

    Returns:
        TranscribeAudioResult containing transcriptions and metadata

    Example: transcribe_with_vad("speech.mp3", target_language="en")
    """
    console.print("[bold blue]Transcribing audio from source[/bold blue]")

    # Step 1: Segment audio using VAD
    console.print("[yellow]Segmenting audio using Voice Activity Detection...[/yellow]")
    audio_stream = audio_from_source(audio_source)
    segments = chunk_by_vad_stream(
        audio_stream, min_speech_duration=Duration.from_ms(10000)
    )

    # Step 2: Transcribe each segment using LINQ with progress tracking
    def transcribe_segment_with_index(indexed_segment):
        index, segment = indexed_segment
        print(segment.timestamp)
        wav_bytes = chunk_to_wav_bytes(segment)
        transcription_result = transcribe_audio(
            wav_bytes,
            source_language=source_language,
            target_language=target_language,
        )
        segment_transcription = SegmentTranscription(
            segment_index=index,
            start_time=segment.timestamp.as_sec(),
            result=transcription_result,
        )
        return segment_transcription

    # Process segments with enhanced LINQ pipeline
    successful_transcriptions, failed_transcriptions = (
        from_iterable(enumerate(segments))
        .with_progress("Transcribing segments")
        .try_select(transcribe_segment_with_index)
    )

    all_transcriptions = list(successful_transcriptions)
    # Extract all words using LINQ select_many for flattening
    all_words = (
        from_iterable(all_transcriptions)
        .select_many(lambda transcription: transcription.result.words)
        .to_list()
    )

    # Report any transcription failures
    for failure in failed_transcriptions:
        traceback.print_exception(failure)
        console.print(f"Failed to transcribe {failure}")

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
) -> Table:
    """Find new vocabulary words not in database.

    Args:
        transcription_file: Path to transcription JSON file
        output: Output file for new words

    Returns:
        Table containing new vocabulary words

    Example: diff_vocab(Path("transcriptions.json"))
    """
    if not transcription_file.exists():
        raise FileNotFoundError(f"Transcription file not found: {transcription_file}")

    console.print("[bold blue]Comparing vocabulary against database...[/bold blue]")

    # Load transcription data
    with open(transcription_file) as f:
        data = json.load(f)

    # Extract all words using LINQ select_many for flattening
    new_words, existing_words = (
        from_iterable(data["transcriptions"])
        .select_many(lambda transcription: transcription["result"]["words"])
        .where(lambda word: word["word_native"] and word["word_translated"])
        .select(
            lambda word: TranscribedWord(
                word_native=word["word_native"],
                word_translated=word["word_translated"],
            )
        )
        .partition(lambda word: len(vocab_search(word=word.word_native, limit=1)) == 0)
    )

    existing_count = sum(1 for _ in existing_words)
    new_words = list(new_words)

    console.print(f"[green]Found {len(new_words)} new words[/green]")
    console.print(f"[yellow]{existing_count} words already in database[/yellow]")

    if output:
        with open(output, "w") as f:
            json.dump(to_json_dict(new_words), f, indent=2, ensure_ascii=False)
        console.print(f"[green]New words saved to:[/green] {output}")

    return Table.from_pydantic(new_words)


class ExportCsvResult(BaseModel):
    """Result of CSV export."""

    exported_count: int
    output_file: str


@register()
def export_csv(
    new_words_table: Enumerable[TranscribedWord],
    output_csv: Path,
) -> ExportCsvResult:
    """Export new vocabulary words to CSV.

    Args:
        new_words_table: Table containing new vocabulary words
        output_csv: Output CSV file path

    Returns:
        ExportCsvResult containing export counts and success status

    Example: export_csv(vocab_table, Path("vocabulary.csv"))
    """
    console.print(f"[bold blue]Exporting to CSV:[/bold blue] {output_csv}")
    new_words_table = new_words_table.to_table()

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ['word', 'translation', 'source_language', 'target_language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for vocab_word in new_words_table:
            writer.writerow(
                {
                    "word": vocab_word.word_native,
                    "translation": vocab_word.word_translated,
                    "source_language": "auto-detected",
                    "target_language": "en",
                }
            )

    console.print(f"[green]Exported {len(new_words_table)} words to CSV[/green]")
    return ExportCsvResult(
        exported_count=len(new_words_table), output_file=str(output_csv)
    )


@register()
def full_pipeline(
    audio_source: SourceLike,
    source_language: str | None = None,
    target_language: str = "en",
    output_dir: Path | None = None,
    auto_add: bool = False,
) -> FullPipelineResult:
    """Run complete transcription and vocabulary extraction pipeline.

    Args:
        audio_source: Audio source (file path, bytes, URL, etc.)
        source_language: Source language
        target_language: Target language
        output_dir: Output directory for intermediate files
        auto_add: Automatically add all new words without review

    Returns:
        FullPipelineResult containing pipeline statistics

    Example: full_pipeline("speech.mp3", auto_add=True)
    """
    console.print("[bold blue]Running full pipeline on audio source[/bold blue]")

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
    transcribe_result = transcribe_with_vad(
        audio_source=audio_source,
        source_language=source_language,
        target_language=target_language,
        output=transcription_file,
    )

    # Step 2: Find new vocabulary
    console.print("\n[bold]Step 2: Finding new vocabulary[/bold]")
    new_words_table = diff_vocab(
        transcription_file=transcription_file,
        output=new_words_file,
    )

    # Step 3: Export to CSV
    console.print("\n[bold]Step 3: Exporting to CSV[/bold]")
    export_csv(
        new_words_table=new_words_table,
        output_csv=csv_file,
    )

    # Step 4: Add vocabulary to database
    console.print("\n[bold]Step 4: Adding vocabulary to database[/bold]")
    if auto_add:
        selected_words = new_words_table
    else:
        selected_words = select_ui(
            new_words_table,
            title="New Vocabulary Words - Select words to add",
        )

    selected_words = selected_words.to_table()

    for vocab_word in selected_words.with_progress():
        vocab_add(
            word=vocab_word.word,
            translation=vocab_word.translation,
            tags=["transcribed"],
        )

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"[green]Results saved in:[/green] {output_dir}")

    return FullPipelineResult(
        total_segments=transcribe_result.total_segments,
        total_words=transcribe_result.total_words,
        new_words=len(selected_words),
    )


if __name__ == "__main__":
    functions = [
        transcribe_with_vad,
        diff_vocab,
        export_csv,
        full_pipeline,
    ]
    cli_main(
        functions,
        context_cls=ToolContext,
    )

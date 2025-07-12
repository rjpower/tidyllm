#!/usr/bin/env python3
"""Long-form audio transcription and vocabulary extraction tools.

This module provides tools for processing audio files through VAD segmentation,
transcription, vocabulary extraction, and interactive review for adding new
words to the vocabulary database.
"""

import csv
import tempfile
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console

from tidyllm.adapters.cli import cli_main
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext
from tidyllm.tools.transcribe import (
    TranscribedWord,
    TranscriptionResult,
    transcribe_audio,
)
from tidyllm.tools.vad import (
    chunk_by_vad_stream,
)
from tidyllm.tools.vocab_table import vocab_add, vocab_search
from tidyllm.types.duration import Duration
from tidyllm.types.linq import Enumerable, Table
from tidyllm.types.part import Part
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
    source_language: str
    target_language: str


class FullPipelineResult(BaseModel):
    """Result of full pipeline."""

    new_words: int


@register()
def transcribe_with_vad(
    audio_url: str,
    source_language: str | None = None,
    target_language: str = "en",
) -> TranscribeAudioResult:
    """Transcribe audio from URL with VAD segmentation and extract vocabulary.

    Args:
        audio_url: Audio URL (file://, audio://, mic://, etc.)
        source_language: Source language (auto-detect if not provided)
        target_language: Target language for translation

    Returns:
        TranscribeAudioResult containing transcriptions and metadata

    Example: transcribe_with_vad("file://speech.mp3", target_language="en")
    """
    console.print("[bold blue]Transcribing audio from URL[/bold blue]")

    # Step 1: Segment audio using VAD
    console.print("[yellow]Segmenting audio using Voice Activity Detection...[/yellow]")
    audio_stream = Part.from_url(audio_url)
    segments = chunk_by_vad_stream(
        audio_stream, min_speech_duration=Duration.from_ms(10000)
    )

    # Step 2: Transcribe each segment using LINQ with progress tracking
    def transcribe_segment_with_index(indexed_segment):
        index, segment = indexed_segment
        print(segment.timestamp)
        wav_bytes = chunk_to_wav_bytes(segment)
        # Convert bytes to Part for transcription
        wav_part = Part.from_bytes(wav_bytes, "audio/wav")
        transcription_result = transcribe_audio(
            wav_part,
            source_language=source_language,
            target_language=target_language,
        )
        return SegmentTranscription(
            segment_index=index,
            start_time=segment.timestamp.as_sec(),
            result=transcription_result,
        )

    # Process segments with enhanced LINQ pipeline
    successful_transcriptions, failed_transcriptions = (
        Table.from_rows(enumerate(segments))
        .with_progress("Transcribing segments")
        .try_select(transcribe_segment_with_index)
    )

    for failure in failed_transcriptions:
        console.print(f"Failed to transcribe {failure}")

    final_result = TranscribeAudioResult(
        transcriptions=successful_transcriptions.to_list(),
        source_language=source_language or "auto-detected",
        target_language=target_language,
    )

    return final_result


@register()
def diff_vocab(transcriptions_url: str) -> Table[TranscribedWord]:
    """Find new vocabulary words not in database.

    Args:
        transcriptions_url: URL to transcription JSON file

    Returns:
        Table containing new vocabulary words

    Example: diff_vocab("file://transcriptions.json")
    """
    # Load transcription data from URL
    transcription_parts = Part.from_url(transcriptions_url)
    transcription_part = next(iter(transcription_parts))
    data = TranscribeAudioResult.model_validate_json(transcription_part.data)

    new_words, existing_words = (
        Table.from_rows(data.transcriptions)
        .select_many(lambda transcription: transcription.result.words)
        .where(lambda word: word.word_native and word.word_translated)
        .partition(lambda word: len(vocab_search(word=word.word_native, limit=1)) == 0)
    )

    new_words = list(new_words)
    console.print(f"[green]Found {len(new_words)} new words[/green]")
    return Table.from_pydantic(new_words)


@register()
def export_csv(
    new_words_table: Enumerable[TranscribedWord],
    output_csv: Path,
):
    """Export new vocabulary words to CSV.

    Args:
        new_words_table: Table containing new vocabulary words
        output_csv: Output CSV file path

    Returns:
        ExportCsvResult containing export counts and success status

    Example: export_csv(vocab_table, Path("vocabulary.csv"))
    """
    new_words_table = new_words_table.materialize()

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


@register()
def full_pipeline(
    audio_url: str,
    source_language: str | None = None,
    target_language: str = "en",
    auto_add: bool = False,
) -> FullPipelineResult:
    """Run complete transcription and vocabulary extraction pipeline.

    Args:
        audio_url: Audio URL (file://, audio://, mic://, etc.)
        source_language: Source language
        target_language: Target language
        auto_add: Automatically add all new words without review

    Returns:
        FullPipelineResult containing pipeline statistics

    Example: full_pipeline("file://speech.mp3", auto_add=True)
    """
    console.print("[bold blue]Running full pipeline on audio URL[/bold blue]")

    output_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))

    csv_file = output_dir / "new_words.csv"

    console.print(f"[yellow]Working directory:[/yellow] {output_dir}")

    # Step 1: Transcribe audio with segmentation
    console.print("\n[bold]Step 1: Transcribing audio with VAD segmentation[/bold]")
    transcribe_result = transcribe_with_vad(
        audio_url=audio_url,
        source_language=source_language,
        target_language=target_language,
    )

    # Step 2: Find new vocabulary
    console.print("\n[bold]Step 2: Finding new vocabulary[/bold]")
    # Save transcription result to temporary file for diff_vocab
    transcription_file = output_dir / "transcriptions.json"
    transcription_file.write_text(transcribe_result.model_dump_json())
    new_words_table = diff_vocab(f"file://{transcription_file}")

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

    selected_words = selected_words.materialize()

    for vocab_word in selected_words.with_progress():
        vocab_add(
            word=vocab_word.word_native,
            translation=vocab_word.word_translated,
            tags=["transcribed"],
        )

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"[green]Results saved in:[/green] {output_dir}")

    return FullPipelineResult(
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

"""Integration tests for add_card app."""

import json
import subprocess
import zipfile
from pathlib import Path

import pytest


def extract_json_from_output(stdout: str) -> dict:
    """Extract JSON from command output that may contain other text."""
    stdout = stdout.strip()
    
    # Find the JSON part (starts with { and ends with })
    json_start = stdout.rfind('{')
    if json_start == -1:
        pytest.fail(f"No JSON found in output: {stdout}")
    
    json_output = stdout[json_start:]
    
    try:
        return json.loads(json_output)
    except json.JSONDecodeError:
        pytest.fail(f"Could not parse JSON from output: {json_output}")


def test_add_card_integration_command():
    """Integration test that runs the actual add_card command and checks output."""
    # Run the actual command
    result = subprocess.run([
        "uv", "run", "./apps/add_card.py", "add_card",
        "--deck-name=IntegrationTest",
        "--term-en=test"
    ], 
    capture_output=True, 
    text=True, 
    cwd=Path(__file__).parent.parent
    )
    
    # Check command succeeded
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Parse the JSON output
    data = extract_json_from_output(result.stdout)
    
    # Verify the result structure
    assert data["card_created"] is True
    assert "deck_path" in data
    assert data["deck_path"].endswith(".apkg")
    assert "IntegrationTest" in data["message"]
    
    # Verify the actual deck file exists
    deck_path = Path(data["deck_path"])
    assert deck_path.exists(), f"Deck file not found at {deck_path}"
    
    # Verify it's a valid zip file (Anki package format)
    with zipfile.ZipFile(deck_path, 'r') as zip_file:
        file_list = zip_file.namelist()
        
        # Should contain required Anki files
        assert 'collection.anki2' in file_list
        assert 'media' in file_list
        
        # Check media mapping contains audio files
        media_content = zip_file.read('media').decode('utf-8')
        media_mapping = json.loads(media_content)
        
        # Should have two audio files (English and Japanese)
        assert len(media_mapping) >= 2
        filenames = list(media_mapping.values())
        # Check for content-based hashed audio filenames
        assert any(filename.startswith('audio_') and filename.endswith('.mp3') for filename in filenames)
        assert len([f for f in filenames if f.startswith('audio_') and f.endswith('.mp3')]) >= 2
        
        # Verify audio files actually exist in the package
        for media_id in media_mapping.keys():
            assert media_id in file_list, f"Media file {media_id} not found in package"


def test_add_card_integration_japanese_term():
    """Integration test with Japanese term input."""
    result = subprocess.run([
        "uv", "run", "./apps/add_card.py", "add_card", 
        "--deck-name=JapaneseTest",
        "--term-ja=犬"
    ],
    capture_output=True,
    text=True,
    cwd=Path(__file__).parent.parent
    )
    
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Parse JSON output
    data = extract_json_from_output(result.stdout)
    
    # Verify result
    assert data["card_created"] is True
    assert "JapaneseTest" in data["message"]
    
    # Verify package
    deck_path = Path(data["deck_path"])
    assert deck_path.exists()
    
    with zipfile.ZipFile(deck_path, 'r') as zip_file:
        # Check it has the expected structure
        assert 'collection.anki2' in zip_file.namelist()
        assert 'media' in zip_file.namelist()


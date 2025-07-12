"""Tests for TTS functionality."""

from unittest.mock import Mock, patch

# AudioPart is imported through tidyllm.types.part

from tidyllm.tools.tts import Voice, generate_speech
from tidyllm.types.part import AudioPart


class TestGenerateSpeech:
    """Test TTS speech generation."""
    
    @patch('librosa.load')
    @patch('litellm.speech')
    def test_generate_speech_basic(self, mock_speech, mock_librosa_load):
        """Test basic speech generation."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"fake audio data"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        # Mock librosa to return fake audio data
        import numpy as np
        mock_librosa_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        
        result = generate_speech("Hello world")
        
        # Verify result is an Enumerable of AudioPart
        audio_parts = result.to_list()
        assert len(audio_parts) == 1
        assert isinstance(audio_parts[0], AudioPart)
        assert audio_parts[0].sample_rate == 22050
        assert audio_parts[0].channels == 1
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Hello world",
            voice=Voice.ZEPHYR
        )
    
    @patch('librosa.load')
    @patch('litellm.speech')
    def test_generate_speech_with_voice(self, mock_speech, mock_librosa_load):
        """Test speech generation with specific voice."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"test audio"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        # Mock librosa to return fake audio data
        import numpy as np
        mock_librosa_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        
        result = generate_speech("Test content", voice=Voice.PUCK)
        
        # Verify result is an Enumerable of AudioPart
        audio_parts = result.to_list()
        assert len(audio_parts) == 1
        assert isinstance(audio_parts[0], AudioPart)
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Test content",
            voice=Voice.PUCK
        )
    
    @patch('librosa.load')
    @patch('litellm.speech')
    def test_generate_speech_with_language(self, mock_speech, mock_librosa_load):
        """Test speech generation with language specification."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"japanese audio"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        # Mock librosa to return fake audio data
        import numpy as np
        mock_librosa_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        
        result = generate_speech("こんにちは", language="Japanese")
        
        # Verify result is an Enumerable of AudioPart
        audio_parts = result.to_list()
        assert len(audio_parts) == 1
        assert isinstance(audio_parts[0], AudioPart)
        
        # Verify API call includes language prompt
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Say the following in Japanese: 'こんにちは'",
            voice=Voice.ZEPHYR
        )
    
    @patch('librosa.load')
    @patch('litellm.speech')
    def test_generate_speech_with_custom_model(self, mock_speech, mock_librosa_load):
        """Test speech generation with custom model."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"custom model audio"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        # Mock librosa to return fake audio data
        import numpy as np
        mock_librosa_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        
        result = generate_speech("Custom model test", model="custom/model")
        
        # Verify result is an Enumerable of AudioPart
        audio_parts = result.to_list()
        assert len(audio_parts) == 1
        assert isinstance(audio_parts[0], AudioPart)
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="custom/model",
            input="Custom model test",
            voice=Voice.ZEPHYR
        )
    
    @patch('librosa.load')
    @patch('litellm.speech')
    def test_generate_speech_chunked_response(self, mock_speech, mock_librosa_load):
        """Test speech generation with chunked audio response."""
        # Mock TTS response with multiple chunks
        mock_response = Mock()
        chunk1 = b"chunk1"
        chunk2 = b"chunk2"
        chunk3 = b"chunk3"
        mock_response.iter_bytes.return_value = [chunk1, chunk2, chunk3]
        mock_speech.return_value = mock_response
        
        # Mock librosa to return fake audio data
        import numpy as np
        mock_librosa_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        
        result = generate_speech("Chunked response test")
        
        # Verify result is an Enumerable of AudioPart with combined chunks
        audio_parts = result.to_list()
        assert len(audio_parts) == 1
        assert isinstance(audio_parts[0], AudioPart)
    
    def test_cache_behavior(self):
        """Test that generate_speech function is cached."""
        # Verify it has cache attributes from functools.wraps
        assert hasattr(generate_speech, '__wrapped__')
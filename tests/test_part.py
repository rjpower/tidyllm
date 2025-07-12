"""Tests for Part validation and functionality."""

import base64
import pytest
from pathlib import Path

from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext
from tidyllm.types.part import Part, BasicPart, TextPart, HtmlPart, AudioPart, ImagePart
import numpy as np
from PIL import Image


@pytest.fixture
def tool_context():
    """Provide a fresh ToolContext for each test."""
    return ToolContext()


class TestBasicPart:
    """Test BasicPart functionality for simple data storage."""

    def test_basicpart_from_bytes(self, tool_context):
        """Test creating BasicPart from bytes."""
        with set_tool_context(tool_context):
            data = b"<h1>Hello</h1>"
            part = BasicPart.from_base64(base64.b64encode(data), "text/html")

            assert part.mime_type == "text/html"
            assert part.data == base64.b64encode(data)

    def test_basicpart_creation(self, tool_context):
        """Test creating BasicPart through registry."""
        with set_tool_context(tool_context):
            part_data = {
                "mime_type": "text/plain",
                "data": base64.b64encode(b"Hello World"),
            }
            part = Part.from_value(part_data)

            assert isinstance(part, BasicPart)
            assert part.mime_type == "text/plain"
            assert part.text == "Hello World"

    def test_unregistered_mime_type_fallback(self, tool_context):
        """Test that unregistered mime types fall back to BasicPart."""
        with set_tool_context(tool_context):
            part_data = {"mime_type": "unknown/type", "data": base64.b64encode(b"test")}

            part = Part.from_value(part_data)
            assert isinstance(part, BasicPart)
            assert part.mime_type == "unknown/type"


class TestAudioPart:
    """Test AudioPart functionality."""

    def test_audiopart_from_array(self, tool_context):
        """Test creating AudioPart from numpy array."""
        with set_tool_context(tool_context):
            # Create test audio data
            test_audio = np.random.random(1000).astype(np.float32)
            audio_part = AudioPart.from_array(test_audio, 16000)

            assert audio_part.sample_rate == 16000
            assert audio_part.channels == 1
            assert audio_part.duration > 0
            assert len(audio_part.to_wav_bytes()) > 0

    def test_audiopart_stereo(self, tool_context):
        """Test AudioPart with stereo audio."""
        with set_tool_context(tool_context):
            # Create stereo test data (2 channels, 1000 samples)
            stereo_audio = np.random.random((2, 1000)).astype(np.float32)
            audio_part = AudioPart.from_array(stereo_audio, 44100)

            assert audio_part.channels == 2
            assert audio_part.sample_rate == 44100
            assert len(audio_part.to_wav_bytes()) > 0

    def test_audiopart_serialization(self, tool_context):
        """Test AudioPart serialization and round-trip."""
        with set_tool_context(tool_context):
            # Create test audio
            test_audio = np.random.random(500).astype(np.float32)
            original = AudioPart.from_array(test_audio, 22050)

            # Convert to base64 and back
            b64_data = original.to_base64()
            assert isinstance(b64_data, str)
            assert len(b64_data) > 0

            # Test round-trip through registry
            part_dict = {"mime_type": "audio/wav", "data": b64_data}
            restored = Part.from_value(part_dict)

            assert isinstance(restored, AudioPart)
            assert restored.sample_rate > 0
            assert restored.channels > 0


class TestImagePart:
    """Test ImagePart functionality."""

    def test_imagepart_from_pil(self, tool_context):
        """Test creating ImagePart from PIL Image."""
        with set_tool_context(tool_context):
            # Create test image
            test_image = Image.new("RGB", (100, 50), color="red")
            image_part = ImagePart.from_pil(test_image)

            assert image_part.width == 100
            assert image_part.height == 50
            assert image_part.format == "PNG"
            assert image_part.mode == "RGB"
            assert image_part.aspect_ratio == 2.0

    def test_imagepart_from_array(self, tool_context):
        """Test creating ImagePart from numpy array."""
        with set_tool_context(tool_context):
            # Create RGB test array
            test_array = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
            image_part = ImagePart.from_array(test_array)

            assert image_part.width == 100
            assert image_part.height == 50
            assert image_part.mode == "RGB"

            # Test round-trip to array
            restored_array = image_part.to_array()
            assert restored_array.shape == test_array.shape

    def test_imagepart_grayscale(self, tool_context):
        """Test ImagePart with grayscale image."""
        with set_tool_context(tool_context):
            # Create grayscale test array
            test_array = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
            image_part = ImagePart.from_array(test_array)

            assert image_part.width == 100
            assert image_part.height == 50
            assert image_part.mode == "L"

    def test_imagepart_rgba(self, tool_context):
        """Test ImagePart with RGBA image."""
        with set_tool_context(tool_context):
            # Create RGBA test array
            test_array = np.random.randint(0, 255, (50, 100, 4), dtype=np.uint8)
            image_part = ImagePart.from_array(test_array)

            assert image_part.width == 100
            assert image_part.height == 50
            assert image_part.mode == "RGBA"

    def test_imagepart_resize(self, tool_context):
        """Test image resizing functionality."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (100, 50), color="blue")
            image_part = ImagePart.from_pil(test_image)

            # Test resize
            resized = image_part.resize(200, 100)
            assert resized.width == 200
            assert resized.height == 100
            assert resized.aspect_ratio == 2.0

    def test_imagepart_resize_to_fit(self, tool_context):
        """Test resize to fit functionality."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (100, 50), color="green")
            image_part = ImagePart.from_pil(test_image)

            # Test resize to fit within smaller bounds
            fitted = image_part.resize_to_fit(80, 80)
            assert fitted.width == 80
            assert fitted.height == 40  # Maintains aspect ratio

    def test_imagepart_crop(self, tool_context):
        """Test image cropping functionality."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (100, 50), color="yellow")
            image_part = ImagePart.from_pil(test_image)

            # Test crop
            cropped = image_part.crop(10, 10, 60, 40)
            assert cropped.width == 50
            assert cropped.height == 30

    def test_imagepart_format_conversion(self, tool_context):
        """Test image format conversion."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (50, 50), color="purple")
            image_part = ImagePart.from_pil(test_image, "PNG")

            # Test convert to JPEG
            jpeg_part = image_part.convert_format("JPEG")
            assert jpeg_part.format == "JPEG"
            assert jpeg_part.width == 50
            assert jpeg_part.height == 50

    def test_imagepart_mode_conversion(self, tool_context):
        """Test image mode conversion."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (50, 50), color="orange")
            image_part = ImagePart.from_pil(test_image)

            # Test convert to grayscale
            gray_part = image_part.convert_mode("L")
            assert gray_part.mode == "L"
            assert gray_part.width == 50
            assert gray_part.height == 50

    def test_imagepart_serialization(self, tool_context):
        """Test ImagePart serialization and round-trip."""
        with set_tool_context(tool_context):
            # Create test image
            test_image = Image.new("RGB", (30, 20), color="cyan")
            original = ImagePart.from_pil(test_image)

            # Convert to base64 and back
            b64_data = original.to_base64()
            assert isinstance(b64_data, str)
            assert len(b64_data) > 0

            # Test round-trip through registry
            part_dict = {"mime_type": "image/png", "data": b64_data}
            restored = Part.from_value(part_dict)

            assert isinstance(restored, ImagePart)
            assert restored.width == 30
            assert restored.height == 20

    def test_imagepart_bytes_conversion(self, tool_context):
        """Test ImagePart to bytes conversion."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (40, 30), color="magenta")
            image_part = ImagePart.from_pil(test_image)

            # Test PNG bytes
            png_bytes = image_part.to_bytes("PNG")
            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0

            # Test JPEG bytes
            jpeg_bytes = image_part.to_bytes("JPEG")
            assert isinstance(jpeg_bytes, bytes)
            assert len(jpeg_bytes) > 0

            # Test round-trip
            restored = ImagePart.from_bytes(png_bytes)
            assert restored.width == 40
            assert restored.height == 30

    def test_imagepart_from_bytes(self, tool_context):
        """Test creating ImagePart from bytes."""
        with set_tool_context(tool_context):
            # Create a test PNG image in memory
            test_image = Image.new("RGB", (25, 25), color="brown")
            import io

            buffer = io.BytesIO()
            test_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            # Create ImagePart from bytes
            image_part = ImagePart.from_bytes(image_bytes)
            assert image_part.width == 25
            assert image_part.height == 25
            assert image_part.mode == "RGB"

    def test_imagepart_text_property(self, tool_context):
        """Test ImagePart text property for compatibility."""
        with set_tool_context(tool_context):
            test_image = Image.new("RGB", (80, 60), color="pink")
            image_part = ImagePart.from_pil(test_image, "JPEG")

            # Test that text property returns a descriptive string
            text = image_part.text
            assert isinstance(text, str)
            assert "80" in text
            assert "60" in text
            assert "JPEG" in text
            assert "RGB" in text

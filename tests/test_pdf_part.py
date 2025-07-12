"""Tests for PdfPart."""

import pytest
from pathlib import Path

from tidyllm.types.part import Part, PdfPart, is_pdf_part
from tidyllm.types.part.image import ImagePart


def test_pdf_part_from_file():
    """Test creating PdfPart from file URL."""
    # Use the test PDF file
    pdf_path = Path("docs/tidyllm-spec.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Load PDF from file URL
    pdf_url = f"file://{pdf_path.absolute()}"
    pdf_parts = Part.from_url(pdf_url)
    pdf_part = next(iter(pdf_parts))
    
    # Verify it's a PdfPart
    assert is_pdf_part(pdf_part)
    assert isinstance(pdf_part, PdfPart)
    assert pdf_part.mime_type.startswith("application/pdf")
    assert pdf_part.page_count > 0


def test_pdf_part_extract_images():
    """Test extracting images from PDF."""
    pdf_path = Path("docs/tidyllm-spec.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Load PDF
    pdf_url = f"file://{pdf_path.absolute()}"
    pdf_parts = Part.from_url(pdf_url)
    pdf_part = next(iter(pdf_parts))
    
    assert is_pdf_part(pdf_part)
    
    # Extract images
    images = pdf_part.extract_images(image_width=400, image_height=600, jpeg_quality=80)
    
    # Verify we got ImagePart objects
    assert len(images) == pdf_part.page_count
    assert all(isinstance(img, ImagePart) for img in images)
    assert all(img.width == 400 for img in images)
    assert all(img.height == 600 for img in images)
    assert all(img.mime_type.startswith("image/") for img in images)


def test_pdf_part_as_images():
    """Test PdfPart as_images() method returns Enumerable of ImageParts."""
    pdf_path = Path("docs/tidyllm-spec.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Load PDF
    pdf_url = f"file://{pdf_path.absolute()}"
    pdf_parts = Part.from_url(pdf_url)
    pdf_part = next(iter(pdf_parts))
    
    assert is_pdf_part(pdf_part)
    
    # Get as ImageParts
    image_parts = pdf_part.as_images()
    image_list = image_parts.to_list()
    
    # Verify we got ImageParts
    assert len(image_list) == pdf_part.page_count
    assert all(isinstance(img, ImagePart) for img in image_list)
    assert all(img.mime_type.startswith("image/") for img in image_list)


def test_pdf_part_linq_operations():
    """Test that PdfPart can be used with LINQ operations."""
    pdf_path = Path("docs/tidyllm-spec.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Load PDF
    pdf_url = f"file://{pdf_path.absolute()}"
    pdf_parts = Part.from_url(pdf_url)
    pdf_part = next(iter(pdf_parts))
    
    assert is_pdf_part(pdf_part)
    
    # Use LINQ operations on the images
    image_parts = pdf_part.as_images()
    
    # Test some LINQ operations
    count = image_parts.count()
    assert count == pdf_part.page_count
    
    first_image = image_parts.first()
    assert isinstance(first_image, ImagePart)
    
    # Test filtering (all should be JPEG format)
    jpeg_images = image_parts.where(lambda img: "jpeg" in img.format.lower())
    assert jpeg_images.count() == count
    
    # Test select operation
    sizes = image_parts.select(lambda img: (img.width, img.height)).to_list()
    assert len(sizes) == count
    assert all(isinstance(size, tuple) and len(size) == 2 for size in sizes)


def test_pdf_part_serialization():
    """Test PdfPart serialization and deserialization."""
    pdf_path = Path("docs/tidyllm-spec.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Create PdfPart from raw bytes
    pdf_data = pdf_path.read_bytes()
    pdf_part = PdfPart.from_bytes("application/pdf", pdf_data)
    
    # Test serialization
    serialized = pdf_part.model_dump()
    assert "mime_type" in serialized
    assert "data" in serialized
    assert "page_count" in serialized
    assert "_pdf_doc" not in serialized  # Private attribute should be excluded
    
    # Test deserialization
    pdf_part_2 = PdfPart.model_validate(serialized)
    assert pdf_part_2.page_count == pdf_part.page_count
    assert pdf_part_2.mime_type == pdf_part.mime_type
    
    # Test that PDF document can be accessed after deserialization
    doc = pdf_part_2.pdf_document
    assert doc is not None
    assert len(doc) == pdf_part_2.page_count
    
    # Test JSON round-trip
    json_str = pdf_part.model_dump_json()
    pdf_part_3 = PdfPart.model_validate_json(json_str)
    assert pdf_part_3.page_count == pdf_part.page_count
    assert pdf_part_3.mime_type == pdf_part.mime_type
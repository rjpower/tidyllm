"""PDF-specific Part sources and utilities."""

import base64
import io
from pathlib import Path

import pypdfium2 as pdfium
from pydantic_core import Url

from tidyllm.types.linq import Enumerable, Table
from tidyllm.types.part.image import ImagePart
from tidyllm.types.part.lib import PART_SOURCE_REGISTRY, Part


class PdfPart(Part):
    """PDF Part with native pypdfium2 storage and image extraction capabilities.
    
    A PDF is also an Enumerable over the underlying images extracted from its pages.
    """

    page_count: int

    def model_post_init(self, __context) -> None:
        """Post-initialization to set up PDF data."""
        if not hasattr(self, "_pdf_bytes"):
            self._pdf_bytes = b""
        if not hasattr(self, "_pdf_doc"):
            self._pdf_doc = None

    @classmethod
    def from_bytes(cls, pdf_bytes: bytes) -> "PdfPart":
        """Create PdfPart from PDF bytes."""
        # Load PDF to get page count
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
        page_count = len(pdf_doc)

        # Create mime_type with metadata
        mime_type = f"application/pdf;pages={page_count}"

        # Create instance with model_construct to bypass validation
        instance = cls.model_construct(
            mime_type=mime_type,
            page_count=page_count,
        )
        instance._pdf_bytes = pdf_bytes
        instance._pdf_doc = pdf_doc

        return instance

    @property
    def pdf_document(self) -> pdfium.PdfDocument:
        """Access to pypdfium2 PDF document."""
        if self._pdf_doc is None:
            self._pdf_doc = pdfium.PdfDocument(self._pdf_bytes)
        return self._pdf_doc

    @property
    def bytes(self) -> bytes:
        """Get raw PDF bytes."""
        return self._pdf_bytes

    def extract_images(
        self,
        image_width: int = 768,
        image_height: int = 1084,
        jpeg_quality: int = 85,
    ) -> list[ImagePart]:
        """Extract pages from PDF as ImagePart objects.
        
        Args:
            image_width: Target width for extracted images
            image_height: Target height for extracted images  
            jpeg_quality: JPEG compression quality (1-100)
            
        Returns:
            List of ImagePart objects with page images
        """
        pdf_doc = self.pdf_document
        image_parts = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc.get_page(page_num)

            # Render page as bitmap
            bitmap = page.render(scale=1.0, rotation=0, crop=(0, 0, 0, 0))

            # Convert to PIL Image and resize
            pil_image = bitmap.to_pil()
            pil_image = pil_image.resize((image_width, image_height))

            # Convert to JPEG bytes
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="JPEG", quality=jpeg_quality)
            img_bytes.seek(0)

            # Create ImagePart from bytes
            image_part = ImagePart.from_bytes(img_bytes.getvalue(), format="JPEG")
            image_parts.append(image_part)

        return image_parts

    def as_images(self) -> "Enumerable[ImagePart]":
        """Return an Enumerable over PDF pages as ImagePart objects.
        
        Makes PdfPart compatible with LINQ operations over the underlying images.
        """
        image_parts = self.extract_images()
        return Table.from_rows(image_parts)

    @property
    def text(self) -> str:
        """Return a text description of the PDF for compatibility with TextPart interface."""
        return f"PDF ({self.page_count} pages)"


class PdfPartSource:
    """PartSource implementation that returns PdfPart instances for PDF mime types."""
    
    def from_dict(self, d: dict) -> Part:
        """Create PdfPart from dictionary representation."""
        data_bytes = base64.b64decode(d["data"])
        return PdfPart.from_bytes(data_bytes)


class PdfFileSource:
    """Stream PDF Parts from file sources."""

    def __init__(self, allowed_dirs: list[Path] | None = None):
        self.allowed_dirs = allowed_dirs or [Path(".")]

    def from_url(self, url: Url) -> Enumerable[Part]:
        """Stream PDF Parts from file URL."""
        path = Path(url.path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        resolved_path = path.resolve()
        for allowed_dir in self.allowed_dirs:
            if resolved_path.is_relative_to(allowed_dir.resolve()):
                return self._load_pdf_file(path)

        raise ValueError(f"Path {path} not in allowed directories")

    def from_dict(self, d: dict) -> Part:
        """PdfFileSource doesn't support from_dict - only URL loading."""
        raise NotImplementedError("PdfFileSource only supports from_url, not from_dict")

    def _load_pdf_file(self, path: Path) -> Enumerable[Part]:
        """Load PDF file as PdfPart."""
        pdf_bytes = path.read_bytes()
        pdf_part = PdfPart.from_bytes(pdf_bytes)
        return Table.from_rows([pdf_part])


# Register PdfPart with the Part registry
PART_SOURCE_REGISTRY.register_scheme("pdf", PdfFileSource([Path(".")]))

pdf_part_source = PdfPartSource()
PART_SOURCE_REGISTRY.register_mimetype("application/pdf", pdf_part_source)

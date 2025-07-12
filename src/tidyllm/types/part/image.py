"""Image-specific Part sources and utilities."""

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic_core import Url

from tidyllm.types.linq import Enumerable, Table
from tidyllm.types.part.lib import PART_SOURCE_REGISTRY, Part

# Top-level image utility functions

def resize_image(image: Image.Image, width: int, height: int, resample=None) -> Image.Image:
    """Resize image to specified dimensions."""
    if resample is None:
        resample = Image.Resampling.LANCZOS
    return image.resize((width, height), resample)


def crop_image(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop image to specified box (left, top, right, bottom)."""
    return image.crop(box)


def convert_image_format(image: Image.Image, format: str) -> Image.Image:
    """Convert image to specified format."""
    if format.upper() == "JPEG" and image.mode in ("RGBA", "LA", "P"):
        # Convert to RGB for JPEG
        return image.convert("RGB")
    elif format.upper() == "PNG" and image.mode not in ("RGBA", "RGB", "L", "P"):
        # Convert to RGBA for PNG with transparency
        return image.convert("RGBA")
    return image


def image_to_bytes(image: Image.Image, format: str = "PNG", **kwargs) -> bytes:
    """Convert PIL Image to bytes in specified format."""
    buffer = io.BytesIO()
    
    # Apply format-specific conversions
    converted_image = convert_image_format(image, format)
    
    # Set default quality for JPEG
    if format.upper() == "JPEG" and "quality" not in kwargs:
        kwargs["quality"] = 95
        
    converted_image.save(buffer, format=format.upper(), **kwargs)
    buffer.seek(0)
    return buffer.read()


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Create PIL Image from bytes."""
    return Image.open(io.BytesIO(image_bytes))


class ImagePart(Part):
    """Image Part with native PIL.Image storage and image processing capabilities."""

    width: int
    height: int
    format: str
    mode: str

    def model_post_init(self, _context) -> None:
        """Post-initialization to set up image data."""
        if not hasattr(self, "_image"):
            # Create a default empty image if none provided
            self._image = Image.new("RGB", (1, 1))

    def to_bytes(self, format: str | None = None, **kwargs) -> bytes:
        """Convert PIL Image to bytes in specified format."""
        if not hasattr(self, "_image"):
            return b""
            
        output_format = format or self.format
        return image_to_bytes(self._image, output_format, **kwargs)

    def to_base64(self, format: str | None = None, **kwargs) -> str:
        """Convert to base64-encoded image data for serialization."""
        return base64.b64encode(self.to_bytes(format, **kwargs)).decode()

    @classmethod
    def from_pil(cls, image: Image.Image, format: str | None = None) -> "ImagePart":
        """Create ImagePart from PIL Image."""
        if format is None:
            format = getattr(image, "format", "PNG") or "PNG"
            
        # Create mime_type with metadata
        mime_type = f"image/{format.lower()};width={image.width};height={image.height};mode={image.mode}"
        
        # Create instance with model_construct to bypass validation
        instance = cls.model_construct(
            mime_type=mime_type,
            width=image.width,
            height=image.height,
            format=format.upper(),
            mode=image.mode,
        )
        instance._image = image.copy()
        
        return instance

    @classmethod
    def from_bytes(cls, image_bytes: bytes, format: str | None = None) -> "ImagePart":
        """Create ImagePart from image bytes."""
        image = image_from_bytes(image_bytes)
        detected_format = format or getattr(image, "format", "PNG") or "PNG"
        return cls.from_pil(image, detected_format)

    @classmethod
    def from_array(cls, array: np.ndarray, format: str = "PNG") -> "ImagePart":
        """Create ImagePart from numpy array."""
        if array.ndim == 2:
            # Grayscale
            image = Image.fromarray(array, mode="L")
        elif array.ndim == 3:
            if array.shape[2] == 3:
                # RGB
                image = Image.fromarray(array, mode="RGB")
            elif array.shape[2] == 4:
                # RGBA
                image = Image.fromarray(array, mode="RGBA")
            else:
                raise ValueError(f"Unsupported array shape: {array.shape}")
        else:
            raise ValueError(f"Array must be 2D or 3D, got {array.ndim}D")
            
        return cls.from_pil(image, format)

    @property
    def image(self) -> Image.Image:
        """Access to PIL Image."""
        return self._image

    @property
    def size(self) -> tuple[int, int]:
        """Image size as (width, height)."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 1.0

    def resize(self, width: int, height: int, resample=None) -> "ImagePart":
        """Return new ImagePart with different size."""
        resized_image = resize_image(self._image, width, height, resample)
        return ImagePart.from_pil(resized_image, self.format)

    def resize_to_fit(self, max_width: int, max_height: int, resample=None) -> "ImagePart":
        """Resize maintaining aspect ratio to fit within max dimensions."""
        # Calculate scaling factor
        width_scale = max_width / self.width
        height_scale = max_height / self.height
        scale = min(width_scale, height_scale)
        
        if scale >= 1.0:
            return ImagePart.from_pil(self._image.copy(), self.format)
            
        new_width = int(self.width * scale)
        new_height = int(self.height * scale)
        
        return self.resize(new_width, new_height, resample)

    def crop(self, left: int, top: int, right: int, bottom: int) -> "ImagePart":
        """Return cropped ImagePart."""
        cropped_image = crop_image(self._image, (left, top, right, bottom))
        return ImagePart.from_pil(cropped_image, self.format)

    def convert_format(self, format: str) -> "ImagePart":
        """Return ImagePart converted to different format."""
        converted_image = convert_image_format(self._image, format)
        return ImagePart.from_pil(converted_image, format)

    def convert_mode(self, mode: str) -> "ImagePart":
        """Return ImagePart converted to different color mode."""
        converted_image = self._image.convert(mode)
        return ImagePart.from_pil(converted_image, self.format)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self._image)

    @property
    def text(self) -> str:
        """Return a text description of the image for compatibility with TextPart interface."""
        return f"Image ({self.width}x{self.height}, {self.format}, {self.mode})"


class ImagePartLoader:
    """MimeLoader for image types."""

    def from_json(self, d: dict[str, Any]) -> Part:
        """Create ImagePart from JSON dictionary."""
        return ImagePart.model_validate(d)

    def from_bytes(self, mime_type: str, data: bytes) -> Part:
        """Create ImagePart from raw bytes."""
        # Parse format from mime_type
        format = "PNG"
        if "/" in mime_type:
            format = mime_type.split("/")[1].split(";")[0].upper()

        return ImagePart.from_bytes(data, format=format)


class ImageFileSource:
    """Stream image Parts from file sources."""

    def __init__(self, allowed_dirs: list[Path] | None = None):
        self.allowed_dirs = allowed_dirs or [Path(".")]

    def from_url(self, url: Url) -> Enumerable[Part]:
        """Stream image Parts from file URL."""
        path = Path(url.path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        resolved_path = path.resolve()
        for allowed_dir in self.allowed_dirs:
            if resolved_path.is_relative_to(allowed_dir.resolve()):
                params = self._parse_query_params(url.query)
                return self._load_image_file(path, params)

        raise ValueError(f"Path {path} not in allowed directories")

    def from_dict(self, d: dict) -> Part:
        """ImageFileSource doesn't support from_dict - only URL loading."""
        raise NotImplementedError("ImageFileSource only supports from_url, not from_dict")

    def from_data(self, mimetype: str, data: bytes) -> Part:
        """ImageFileSource doesn't support from_data - only URL loading."""
        raise NotImplementedError("ImageFileSource only supports from_url, not from_data")

    def _parse_query_params(self, query: str | None) -> dict[str, Any]:
        """Parse URL query parameters."""
        params = {}
        if not query:
            return params

        for param in query.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                if key in ["width", "height", "quality"]:
                    params[key] = int(value)
                else:
                    params[key] = value
        return params

    def _load_image_file(self, path: Path, params: dict[str, Any]) -> Enumerable[Part]:
        """Load image file as ImagePart."""
        image_bytes = path.read_bytes()
        image_part = ImagePart.from_bytes(image_bytes)

        # Apply transformations based on query parameters
        if "width" in params and "height" in params:
            image_part = image_part.resize(params["width"], params["height"])
        elif "width" in params:
            # Maintain aspect ratio
            new_height = int(image_part.height * params["width"] / image_part.width)
            image_part = image_part.resize(params["width"], new_height)
        elif "height" in params:
            # Maintain aspect ratio
            new_width = int(image_part.width * params["height"] / image_part.height)
            image_part = image_part.resize(new_width, params["height"])

        if "format" in params:
            image_part = image_part.convert_format(params["format"])

        return Table.from_rows([image_part])


# Register ImagePart with the Part registry
PART_SOURCE_REGISTRY.register_scheme("image", ImageFileSource([Path(".")]))

image_part_loader = ImagePartLoader()
PART_SOURCE_REGISTRY.register_mimetype("image/png", image_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("image/jpeg", image_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("image/jpg", image_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("image/gif", image_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("image/webp", image_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("image/bmp", image_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("image/tiff", image_part_loader)

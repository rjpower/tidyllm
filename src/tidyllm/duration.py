"""Duration type with nanosecond precision for consistent time handling."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Duration(BaseModel):
    """Duration type with nanosecond precision for consistent time handling.
    
    Core representation is int nanoseconds to avoid float precision issues.
    Provides constructors and accessors for common time units.
    """
    
    nanos: int = Field(description="Duration in nanoseconds")
    
    @model_validator(mode='before')
    @classmethod
    def validate_nanos(cls, values):
        if isinstance(values, dict) and 'nanos' in values:
            if values['nanos'] < 0:
                raise ValueError("Duration cannot be negative")
        return values
    
    @classmethod
    def from_ns(cls, nanos: int) -> Duration:
        """Create duration from nanoseconds."""
        return cls(nanos=nanos)
    
    @classmethod
    def from_us(cls, micros: int | float) -> Duration:
        """Create duration from microseconds."""
        return cls(nanos=int(micros * 1_000))
    
    @classmethod
    def from_ms(cls, millis: int | float) -> Duration:
        """Create duration from milliseconds."""
        return cls(nanos=int(millis * 1_000_000))
    
    @classmethod
    def from_sec(cls, seconds: int | float) -> Duration:
        """Create duration from seconds."""
        return cls(nanos=int(seconds * 1_000_000_000))
    
    @classmethod
    def from_samples(cls, samples: int, sample_rate: int) -> Duration:
        """Create duration from sample count and sample rate."""
        seconds = samples / sample_rate
        return cls.from_sec(seconds)
    
    def as_ns(self) -> int:
        """Get duration as nanoseconds."""
        return self.nanos
    
    def as_us(self) -> float:
        """Get duration as microseconds."""
        return self.nanos / 1_000
    
    def as_ms(self) -> float:
        """Get duration as milliseconds."""
        return self.nanos / 1_000_000
    
    def as_sec(self) -> float:
        """Get duration as seconds."""
        return self.nanos / 1_000_000_000
    
    def as_samples(self, sample_rate: int) -> int:
        """Get duration as sample count for given sample rate."""
        return int(self.as_sec() * sample_rate)
    
    def __add__(self, other: Duration) -> Duration:
        """Add two durations."""
        return Duration(nanos=self.nanos + other.nanos)
    
    def __sub__(self, other: Duration) -> Duration:
        """Subtract two durations."""
        result_nanos = self.nanos - other.nanos
        if result_nanos < 0:
            raise ValueError("Duration cannot be negative")
        return Duration(nanos=result_nanos)
    
    def __mul__(self, scalar: int | float) -> Duration:
        """Multiply duration by a scalar."""
        return Duration(nanos=int(self.nanos * scalar))
    
    def __rmul__(self, scalar: int | float) -> Duration:
        """Multiply duration by a scalar (reverse)."""
        return self * scalar
    
    def __truediv__(self, scalar: int | float) -> Duration:
        """Divide duration by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return Duration(nanos=int(self.nanos / scalar))
    
    def __floordiv__(self, other: Duration) -> int:
        """Floor division of two durations."""
        if other.nanos == 0:
            raise ZeroDivisionError("Cannot divide by zero duration")
        return self.nanos // other.nanos
  
    def __lt__(self, other: Duration) -> bool:
        """Less than comparison."""
        return self.nanos < other.nanos
    
    def __le__(self, other: Duration) -> bool:
        """Less than or equal comparison."""
        return self.nanos <= other.nanos
    
    def __gt__(self, other: Duration) -> bool:
        """Greater than comparison."""
        return self.nanos > other.nanos
    
    def __ge__(self, other: Duration) -> bool:
        """Greater than or equal comparison."""
        return self.nanos >= other.nanos
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Duration):
            return False
        return self.nanos == other.nanos
    
    def __ne__(self, other: object) -> bool:
        """Not equal comparison."""
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        """String representation."""
        if self.nanos == 0:
            return "0s"
        elif self.nanos < 1_000:
            return f"{self.nanos}ns"
        elif self.nanos < 1_000_000:
            return f"{self.as_us():.1f}μs"
        elif self.nanos < 1_000_000_000:
            return f"{self.as_ms():.1f}ms"
        else:
            return f"{self.as_sec():.3f}s"
    
    def __repr__(self) -> str:
        """Representation."""
        return f"Duration({self.nanos}ns)"
    
    @classmethod
    def zero(cls) -> Duration:
        """Create zero duration."""
        return cls(nanos=0)
    
    def is_zero(self) -> bool:
        """Check if duration is zero."""
        return self.nanos == 0
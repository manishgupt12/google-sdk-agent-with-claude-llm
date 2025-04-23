# test_agent/models.py
from pydantic import BaseModel, Field
from typing import Literal

# Import from adktools
from adktools.models import DomainError  

# Input model for get_current_time
class GetCurrentTimeInput(BaseModel):
    timezone: str = Field(
        ...,
        description="IANA timezone name (e.g., 'America/New_York', 'Europe/London')"
    )

# Output model for time results
class TimeResult(BaseModel):
    timezone: str = Field(
        ...,
        description="The timezone used for the time calculation"
    )
    datetime: str = Field(
        ...,
        description="Current time in ISO format (e.g., '2025-03-20T15:30:00')"
    )
    is_dst: bool = Field(
        ...,
        description="Indicates if daylight saving time is in effect"
    )

# Domain-specific error models
class InvalidTimezoneError(DomainError):
    """Error model for invalid timezone errors."""
    timezone: str = Field(
        ..., 
        description="The invalid timezone that was provided"
    )
    error_type: Literal["invalid_timezone"] = "invalid_timezone"
    message: str = Field(
        ...,
        description="The error message associated with the invalid timezone"
    )
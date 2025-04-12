# test_agent/tools.py
import datetime
from zoneinfo import ZoneInfo
from pydantic import ValidationError

# adk tools
from adktools import adk_tool

# agent imports
from test_agent.models import GetCurrentTimeInput, TimeResult, InvalidTimezoneError


@adk_tool(
    name="get_time",
    description="Get the current time in a specified timezone. Accepts IANA timezone identifiers."
)
def get_current_time(timezone: str) -> TimeResult | InvalidTimezoneError:
    """Get current time in a specified timezone
    
    Args:
        timezone (str): The timezone for which to retrieve the current time.

    Returns:
        TimeResult: The current time information for the specified timezone.
        InvalidTimezoneError: If the timezone is invalid or cannot be found.
    """
    # Primary validation with domain-specific error handling
    try:
        # Validate the input
        validated_input = GetCurrentTimeInput(timezone=timezone)
    except ValidationError:
        # Return domain-specific error for input validation failures
        return InvalidTimezoneError(
            timezone=timezone,
            message=f"Invalid timezone format: '{timezone}'"
        )
    
    # Timezone resolution with domain-specific error handling
    try:
        # Get the timezone
        tz = ZoneInfo(validated_input.timezone)
    except Exception:
        # Return domain-specific error for timezone not found
        return InvalidTimezoneError(
            timezone=validated_input.timezone,
            message=f"Unknown timezone: '{validated_input.timezone}'"
        )
    
    # Business logic - get current time
    try:
        now = datetime.datetime.now(tz)
        
        # Return the successful result
        return TimeResult(
            timezone=validated_input.timezone,
            datetime=now.isoformat(timespec="seconds"),
            is_dst=bool(now.dst())
        )
    except Exception as e:
        # Let the decorator catch any unexpected errors
        raise RuntimeError(f"Error getting time data: {str(e)}")
    
@adk_tool
def get_timezones_list() -> dict:
    """Returns a list of common IANA timezones grouped by region."""
    timezones = {
        "North America": ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"],
        "Europe": ["Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Rome"],
        "Asia": ["Asia/Tokyo", "Asia/Singapore", "Asia/Dubai", "Asia/Shanghai"],
        "Australia": ["Australia/Sydney", "Australia/Melbourne", "Australia/Perth"]
    }
    return timezones
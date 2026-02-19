#!/usr/bin/env python3
"""
Example tools for Cactus CLI tool calling.

Each function becomes a tool the LLM can call.
Type hints and docstrings are used to generate JSON schemas.
"""

import random
import datetime
from typing import Literal


def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get the current weather for a location.

    Args:
        location: The city name or location to get weather for
        unit: Temperature unit to use

    Returns:
        Weather information including temperature and conditions
    """
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "foggy"]
    temp_c = random.randint(-10, 35)
    temp_f = int(temp_c * 9/5 + 32)

    return {
        "location": location,
        "temperature": temp_c if unit == "celsius" else temp_f,
        "unit": unit,
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 90),
        "timestamp": datetime.datetime.now().isoformat()
    }


def roll_dice(num_dice: int = 1, sides: int = 6) -> dict:
    """Roll one or more dice and return the results.

    Args:
        num_dice: Number of dice to roll (1-10)
        sides: Number of sides on each die (2-100)

    Returns:
        The individual rolls and their sum
    """
    if num_dice < 1 or num_dice > 10:
        return {"error": "num_dice must be between 1 and 10"}
    if sides < 2 or sides > 100:
        return {"error": "sides must be between 2 and 100"}

    rolls = [random.randint(1, sides) for _ in range(num_dice)]

    return {
        "num_dice": num_dice,
        "sides": sides,
        "rolls": rolls,
        "total": sum(rolls),
        "average": sum(rolls) / len(rolls)
    }


def get_time(timezone: str = "UTC") -> dict:
    """Get the current time in a specific timezone.

    Args:
        timezone: Timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London')

    Returns:
        Current time information for the timezone
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)

        return {
            "timezone": timezone,
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "full_datetime": now.isoformat(),
            "day_of_week": now.strftime("%A"),
            "unix_timestamp": int(now.timestamp())
        }
    except ImportError:
        now = datetime.datetime.now()
        return {
            "timezone": "local",
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "full_datetime": now.isoformat(),
            "day_of_week": now.strftime("%A"),
            "unix_timestamp": int(now.timestamp()),
            "note": "pytz not installed, using local time"
        }


TOOLS = {
    "get_weather": get_weather,
    "roll_dice": roll_dice,
    "get_time": get_time
}

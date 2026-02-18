# Cactus Tool Calling

This directory contains the tool calling infrastructure for Cactus CLI.

## Quick Start

```bash
# Build Cactus
cactus build

# Run with example tools
cactus run LiquidAI/LFM2-1.2B --tools python/tools/example_tools.py
```

## How It Works

1. **Define tools** in a Python module (see `example_tools.py`)
2. **Flask server** starts automatically and exposes tools via HTTP
3. **LLM** can call tools during conversation
4. **Results** are automatically fed back to continue generation

## Architecture

```
chat.cpp (C++) <--HTTP--> tools_server.py <--> your_tools.py (Python)
```

- `chat.cpp`: Main chat interface, uses libcurl for HTTP
- `tools_server.py`: Flask server that executes tools
- `example_tools.py`: Example tool implementations

## Creating Your Own Tools

Create a Python file with your tools:

```python
# my_tools.py
from typing import Literal

def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get the current weather for a location.

    Args:
        location: City name
        unit: Temperature unit
    """
    # Your implementation
    return {"temperature": 72, "condition": "sunny"}

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Your implementation
    return eval(expression)

# Registry (required)
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate
}
```

**Important**:
- Type hints are used to generate JSON schemas
- Docstrings become tool descriptions
- Must include `TOOLS` dictionary at the end

## Usage

```bash
# Basic usage
cactus run <model> --tools path/to/your_tools.py

# Example
cactus run LiquidAI/LFM2-1.2B --tools python/tools/example_tools.py
```

## Example Conversation

```
You: What's the weather in Tokyo?

[Tool Call] get_weather
  Args: {"location":"Tokyo","unit":"celsius"}
  Result: {"success":true,"result":{"location":"Tokyo","temperature":18,...}}

Assistant (with tool results): The current weather in Tokyo is 18°C and partly cloudy.
```

## Requirements

- Flask: `pip install flask`
- Optional: `pip install pytz` (for timezone support in example tools)

## Available Example Tools

1. **get_weather(location, unit)** - Get weather info (simulated)
2. **roll_dice(num_dice, sides)** - Roll dice
3. **get_time(timezone)** - Get current time in a timezone

## Technical Details

- Server runs on `http://127.0.0.1:8765` by default
- Tools execute in Python venv with access to installed packages
- Automatic schema generation from type hints
- OpenAI-compatible JSON format
- No security sandbox (tools run with your privileges)

## Troubleshooting

**Server won't start:**
- Check Flask is installed: `pip install flask`
- Port 8765 might be in use

**Tools not working:**
- Verify `TOOLS` dictionary exists in your module
- Check tool functions have proper type hints
- Look for errors in server output

**Type hints:**
- Use `Literal["a", "b"]` for enums
- Basic types: `str`, `int`, `float`, `bool`, `dict`, `list`
- Default values make parameters optional

## Security Note

Tools run with your user privileges. Only use trusted tool files, just like running any Python script.

#!/usr/bin/env python3
"""
Quick test script to verify tools are working.
Run this to check your tool setup before using with Cactus.
"""

import sys
import json
from pathlib import Path

# Test importing tools
try:
    from example_tools import TOOLS, get_weather, roll_dice, get_time
    print("✓ Tools imported successfully")
    print(f"  Found {len(TOOLS)} tools: {', '.join(TOOLS.keys())}")
except ImportError as e:
    print(f"✗ Failed to import tools: {e}")
    sys.exit(1)

# Test schema generation
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from tools_server import generate_tool_schema

    print("\n✓ Schema generation working")
    for name, func in TOOLS.items():
        schema = generate_tool_schema(func)
        print(f"\n{name}:")
        print(json.dumps(schema, indent=2))

except Exception as e:
    print(f"\n✗ Schema generation failed: {e}")
    sys.exit(1)

# Test tool execution
print("\n" + "="*50)
print("Testing tool execution:")
print("="*50)

try:
    print("\n1. Testing get_weather('Tokyo', 'celsius')...")
    result = get_weather("Tokyo", "celsius")
    print(f"   Result: {result}")
    assert "location" in result
    assert "temperature" in result
    print("   ✓ Passed")

    print("\n2. Testing roll_dice(2, 6)...")
    result = roll_dice(2, 6)
    print(f"   Result: {result}")
    assert "rolls" in result
    assert len(result["rolls"]) == 2
    print("   ✓ Passed")

    print("\n3. Testing get_time('UTC')...")
    result = get_time("UTC")
    print(f"   Result: {result}")
    assert "time" in result
    assert "timezone" in result
    print("   ✓ Passed")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Check Flask
print("\n" + "="*50)
print("Checking dependencies:")
print("="*50)

try:
    import flask
    print(f"✓ Flask installed (version {flask.__version__})")
except ImportError:
    print("✗ Flask not installed")
    print("  Install with: pip install flask")
    sys.exit(1)

try:
    import pytz
    print("✓ pytz installed (optional, for timezone support)")
except ImportError:
    print("⚠ pytz not installed (optional)")
    print("  Install with: pip install pytz")

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
print("\nYou can now run:")
print("  cactus run <model> --tools python/tools/example_tools.py")

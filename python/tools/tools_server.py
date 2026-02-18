#!/usr/bin/env python3
"""
Tool execution server for Cactus CLI.

Starts an HTTP server that:
1. Serves tool schemas (JSON format for LLM)
2. Executes tools when requested by chat.cpp
"""

import sys
import json
import inspect
import logging
from typing import get_type_hints, get_origin, get_args
from flask import Flask, request, jsonify

# Import tools from the user-provided module
def load_tools_module(module_path):
    """Dynamically load tools from a Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_tools", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TOOLS


def python_type_to_json_type(py_type) -> str:
    """Convert Python type hint to JSON schema type."""
    origin = get_origin(py_type)

    # Handle Literal types
    if origin is type(Literal):
        return "string"

    # Handle basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array"
    }

    return type_map.get(py_type, "string")


def extract_enum_values(py_type) -> list:
    """Extract enum values from Literal type hints."""
    origin = get_origin(py_type)
    if origin is type(Literal):
        return list(get_args(py_type))
    return None


def generate_tool_schema(func) -> dict:
    """Generate OpenAI-compatible JSON schema from a Python function."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""

    # Extract description (first line of docstring)
    description = doc.split('\n')[0] if doc else func.__name__

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'cls']:
            continue

        param_type = hints.get(param_name, str)
        json_type = python_type_to_json_type(param_type)

        prop = {
            "type": json_type,
            "description": f"The {param_name} parameter"
        }

        # Add enum values if Literal type
        enum_values = extract_enum_values(param_type)
        if enum_values:
            prop["enum"] = enum_values

        properties[param_name] = prop

        # Mark as required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


# Import Literal for type checking
try:
    from typing import Literal
except ImportError:
    # Python 3.7 compatibility
    try:
        from typing_extensions import Literal
    except ImportError:
        Literal = None


def create_app(tools_dict):
    """Create Flask app with tool execution endpoints."""
    app = Flask(__name__)

    # Disable Flask's default logging spam
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/schemas', methods=['GET'])
    def get_schemas():
        """Return OpenAI-compatible tool schemas."""
        schemas = [generate_tool_schema(func) for func in tools_dict.values()]
        return jsonify(schemas)

    @app.route('/execute', methods=['POST'])
    def execute_tool():
        """Execute a tool and return the result."""
        try:
            data = request.json
            tool_name = data.get('name')
            arguments = data.get('arguments', {})

            if tool_name not in tools_dict:
                return jsonify({
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }), 404

            # Execute the tool
            tool_func = tools_dict[tool_name]
            result = tool_func(**arguments)

            return jsonify({
                "success": True,
                "result": result
            })

        except TypeError as e:
            return jsonify({
                "success": False,
                "error": f"Invalid arguments: {str(e)}"
            }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }), 500

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "tools": list(tools_dict.keys())})

    return app


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools_server.py <tools_module.py> [port]", file=sys.stderr)
        sys.exit(1)

    tools_module_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

    # Load tools
    try:
        tools_dict = load_tools_module(tools_module_path)
        print(f"[Tools Server] Loaded {len(tools_dict)} tools: {', '.join(tools_dict.keys())}", file=sys.stderr)
    except Exception as e:
        print(f"[Tools Server] Failed to load tools: {e}", file=sys.stderr)
        sys.exit(1)

    # Create and run app
    app = create_app(tools_dict)
    print(f"[Tools Server] Starting on port {port}...", file=sys.stderr)

    # Run with minimal logging
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()

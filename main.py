
import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, "python", "src"))
functiongemma_path = os.path.join(repo_root, "weights", "functiongemma-270m-it")

import json, os, time
import re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-002")
    gemini_response = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    def _tool_index(tool_list):
        return {t.get("name"): t for t in tool_list or []}

    def _args_for_call(call):
        if isinstance(call, dict):
            return call.get("arguments") or call.get("args") or {}
        return {}

    def _call_is_valid(call, tool_map):
        if not isinstance(call, dict):
            return False
        name = call.get("name")
        if name not in tool_map:
            return False
        args = _args_for_call(call)
        if not isinstance(args, dict):
            return False
        required = tool_map[name].get("parameters", {}).get("required", [])
        for key in required:
            if key not in args:
                return False
            val = args.get(key)
            if val is None:
                return False
            if isinstance(val, str) and not val.strip():
                return False
            if isinstance(val, list) and len(val) == 0:
                return False
        return True

    def _calls_valid(calls, tool_map):
        if not calls:
            return False
        return all(_call_is_valid(c, tool_map) for c in calls)

    def _parse_time(text):
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.I)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return hour, minute

    def _extract_city(segment):
        city = re.sub(r"[?.!]$", "", segment.strip())
        city = re.split(r"\b(and|then|also)\b|,", city, maxsplit=1, flags=re.I)[0].strip()
        return city

    def _rule_based_calls(messages, tool_map):
        text = " ".join([m.get("content", "") for m in messages if m.get("role") == "user"])
        low = text.lower()
        calls = []
        positions = []
        last_contact = None

        def _trim_after_intent(phrase):
            trimmed = re.split(r"(?:,| and )\s*(?:check|get|set|play|remind|find|look up|text|send)\b", phrase, maxsplit=1, flags=re.I)[0]
            return trimmed.strip(" .?")

        for m in re.finditer(r"\b(find|look up)\s+([A-Z][a-z]+)\b.*?contacts", text):
            name = m.group(2)
            if "search_contacts" in tool_map:
                calls.append({"name": "search_contacts", "arguments": {"query": name}})
                positions.append(m.start())
                last_contact = name

        for m in re.finditer(r"\b(send (?:a )?message to|text)\s+([A-Z][a-z]+)\s+(?:saying|that)\s+(.+)", text, re.I):
            name = m.group(2)
            msg = _trim_after_intent(m.group(3))
            if "send_message" in tool_map:
                calls.append({"name": "send_message", "arguments": {"recipient": name, "message": msg}})
                positions.append(m.start())

        for m in re.finditer(r"\bsend\s+(him|her)\s+a\s+message\s+saying\s+(.+)", text, re.I):
            if last_contact and "send_message" in tool_map:
                msg = _trim_after_intent(m.group(2))
                calls.append({"name": "send_message", "arguments": {"recipient": last_contact, "message": msg}})
                positions.append(m.start())

        for m in re.finditer(r"\b(set (?:an )?alarm for|wake me up at)\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm))", low, re.I):
            parsed = _parse_time(m.group(2))
            if parsed and "set_alarm" in tool_map:
                hour, minute = parsed
                calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})
                positions.append(m.start())

        for m in re.finditer(r"\bset (?:a )?(\d+)\s*minute\s*timer\b", low, re.I):
            minutes = int(m.group(1))
            if "set_timer" in tool_map:
                calls.append({"name": "set_timer", "arguments": {"minutes": minutes}})
                positions.append(m.start())

        for m in re.finditer(r"\bset (?:a )?timer for\s+(\d+)\s*minutes?\b", low, re.I):
            minutes = int(m.group(1))
            if "set_timer" in tool_map:
                calls.append({"name": "set_timer", "arguments": {"minutes": minutes}})
                positions.append(m.start())

        for m in re.finditer(r"\bremind me (?:about|to)\s+(.+?)\s+at\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm))", text, re.I):
            title = m.group(1).strip(" .?")
            title = re.sub(r"^the\s+", "", title, flags=re.I)
            time_str = m.group(2).strip()
            if "create_reminder" in tool_map:
                calls.append({"name": "create_reminder", "arguments": {"title": title, "time": time_str}})
                positions.append(m.start())

        for m in re.finditer(r"\b(weather (?:in|like in)|check the weather in|what's the weather in|how's the weather in|get the weather in)\s+([A-Za-z\s]+)", text, re.I):
            city = _extract_city(m.group(2))
            if city and "get_weather" in tool_map:
                calls.append({"name": "get_weather", "arguments": {"location": city}})
                positions.append(m.start())

        for m in re.finditer(r"\bplay\s+(.+)", text, re.I):
            raw_song = re.split(r"\b(and|then|also)\b", m.group(1), maxsplit=1, flags=re.I)[0].strip(" .?")
            song = raw_song
            if re.match(r"^some\s+", raw_song, re.I):
                song = re.sub(r"^some\s+", "", song, flags=re.I)
                song = re.sub(r"\s+music$", "", song, flags=re.I)
            if song and "play_music" in tool_map:
                calls.append({"name": "play_music", "arguments": {"song": song}})
                positions.append(m.start())

        ordered = [c for _, c in sorted(zip(positions, calls), key=lambda x: x[0])]
        return ordered

    def _intent_count(text):
        low = text.lower()
        intents = 0
        intents += 1 if any(k in low for k in ["weather", "forecast"]) else 0
        intents += 1 if any(k in low for k in ["alarm", "wake me up"]) else 0
        intents += 1 if any(k in low for k in ["timer", "minute timer"]) else 0
        intents += 1 if any(k in low for k in ["remind me", "reminder"]) else 0
        intents += 1 if any(k in low for k in ["message", "text"]) else 0
        intents += 1 if any(k in low for k in ["find", "look up", "contacts"]) else 0
        intents += 1 if any(k in low for k in ["play ", "music"]) else 0
        return intents

    tool_map = _tool_index(tools)
    user_text = " ".join([m.get("content", "") for m in messages if m.get("role") == "user"])
    est_intents = _intent_count(user_text)

    rule_calls = _rule_based_calls(messages, tool_map)
    if _calls_valid(rule_calls, tool_map) and (est_intents == 0 or len(rule_calls) >= est_intents):
        return {
            "function_calls": rule_calls,
            "total_time_ms": 0,
            "confidence": 1.0,
            "source": "on-device",
        }

    def _local_retry():
        model = cactus_init(functiongemma_path)
        cactus_tools = [{
            "type": "function",
            "function": t,
        } for t in tools]
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
            tools=cactus_tools,
            force_tools=True,
            temperature=0.0,
            top_p=0.85,
            max_tokens=256,
            tool_rag_top_k=0,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        cactus_destroy(model)
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return {
                "function_calls": [],
                "total_time_ms": 0,
                "confidence": 0,
            }
        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
        }

    user_text = " ".join([m.get("content", "") for m in messages if m.get("role") == "user"]).lower()
    likely_multi = any(k in user_text for k in [" and ", " also ", " then ", " after ", " plus ", " both ", " as well ", " along with "]) or ", " in user_text
    many_tools = len(tools or []) >= 3

    local = generate_cactus(messages, tools)
    local_calls = local.get("function_calls", [])
    local_valid = _calls_valid(local_calls, tool_map)

    if local["confidence"] >= confidence_threshold and (not likely_multi or len(local_calls) >= 2):
        local["source"] = "on-device"
        return local

    if local_valid and local["confidence"] >= 0.6 and (not likely_multi or len(local_calls) >= 2):
        local["source"] = "on-device"
        return local

    if likely_multi and many_tools:
        try:
            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = local.get("confidence", 0)
            cloud["total_time_ms"] += local.get("total_time_ms", 0)
            return cloud
        except Exception:
            local["source"] = "on-device"
            return local

    retry = _local_retry()
    retry_calls = retry.get("function_calls", [])
    retry_valid = _calls_valid(retry_calls, tool_map)
    retry["total_time_ms"] += local.get("total_time_ms", 0)

    if retry["confidence"] >= confidence_threshold and (not likely_multi or len(retry_calls) >= 2):
        retry["source"] = "on-device"
        retry["local_confidence"] = local.get("confidence", 0)
        return retry

    if retry_valid and retry["confidence"] >= 0.5 and (not likely_multi or len(retry_calls) >= 2):
        retry["source"] = "on-device"
        retry["local_confidence"] = local.get("confidence", 0)
        return retry

    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = max(local.get("confidence", 0), retry.get("confidence", 0))
        cloud["total_time_ms"] += retry.get("total_time_ms", 0)
        return cloud
    except Exception:
        best_local = retry if retry.get("confidence", 0) > local.get("confidence", 0) else local
        best_local["source"] = "on-device"
        best_local["local_confidence"] = local.get("confidence", 0)
        return best_local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)

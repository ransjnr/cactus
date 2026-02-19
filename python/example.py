#!/usr/bin/env python3
"""
Cactus Python FFI Example

Usage:
  1. cactus build
  2. cactus download LiquidAI/LFM2-VL-450M
  3. cactus download openai/whisper-small
  4. cd tools && python example.py
"""

import sys
import json
import os
import ssl
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from cactus import (
    cactus_init,
    cactus_complete,
    cactus_transcribe,
    cactus_embed,
    cactus_image_embed,
    cactus_audio_embed,
    cactus_reset,
    cactus_destroy,
)

WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"


def call_external_cloud_text(messages):
    api_key = os.environ.get("CACTUS_CLOUD_API_KEY")
    if not api_key:
        return None

    ssl_ctx = ssl._create_unverified_context()

    prompt = "\n".join(f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages)
    payload = {
        "text": prompt,
        "language": "en-US",
        "model": "gemini-2.5-flash",
    }
    endpoint = f"https://104.198.76.3/api/v1/text"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("text", "")


def complete_with_simple_handoff(model, messages, confidence_threshold=0.7):
    result = json.loads(cactus_complete(
        model,
        json.dumps(messages),
        confidence_threshold=confidence_threshold,
        auto_handoff=False,
    ))

    if result.get("cloud_handoff"):
        try:
            cloud_text = call_external_cloud_text(messages)
            if cloud_text:
                return cloud_text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"[simple-handoff] cloud request failed: {e}")

    return result.get("response") or result.get("local_output") or ""

# Load model
print("Loading LFM2-VL-450M...")
vlm = cactus_init(str(WEIGHTS_DIR / "lfm2-vl-450m"))

# Text completion
messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
response = cactus_complete(vlm, messages)
print("\nCompletion:")
print(json.dumps(json.loads(response), indent=2))

# Simple custom handoff example as a function call
policy_messages = [{"role": "user", "content": "What is 2+2?"}]
final_output = complete_with_simple_handoff(vlm, policy_messages, confidence_threshold=0.7)
print("\nCustom policy final output:")
print(final_output)

# Text embedding
embedding = cactus_embed(vlm, "Hello world")
print(f"\nText embedding dim: {len(embedding)}")

# Image embedding
embedding = cactus_image_embed(vlm, str(ASSETS_DIR / "test_monkey.png"))
print(f"\nImage embedding dim: {len(embedding)}")

# VLM - describe image
messages = json.dumps([{"role": "user", "content": "Describe this image", "images": [str(ASSETS_DIR / "test_monkey.png")]}])
response = cactus_complete(vlm, messages)
print("\nVLM Image Description:")
print(json.dumps(json.loads(response), indent=2))

cactus_reset(vlm)
cactus_destroy(vlm)

# Transcription
print("\nLoading whisper-small...")
whisper = cactus_init(str(WEIGHTS_DIR / "whisper-small"))
whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
response = cactus_transcribe(whisper, str(ASSETS_DIR / "test.wav"), prompt=whisper_prompt)
print("Transcription:")
print(json.dumps(json.loads(response), indent=2))

# Audio embedding
embedding = cactus_audio_embed(whisper, str(ASSETS_DIR / "test.wav"))
print(f"\nAudio embedding dim: {len(embedding)}")

cactus_destroy(whisper)

print("\nDone!")

"""
Cactus FFI Python Bindings

Python bindings for Cactus Engine via FFI. Provides access to:
- Text completion with LLMs (including cloud handoff detection)
- Audio transcription with Whisper models
- Text, image, and audio embeddings
- RAG (Retrieval-Augmented Generation) queries
- Tool RAG (automatic tool selection based on query relevance)
- Streaming transcription

Response Format:
All completion responses use a unified JSON format with all fields always present:
{
    "success": bool,        # True if generation succeeded
    "error": str|null,      # Error message if failed, null otherwise
    "cloud_handoff": bool,  # True if model recommends deferring to cloud
    "response": str|null,   # Generated text, null if cloud_handoff or error
    "function_calls": [],   # List of function calls if tools were used
    "confidence": float,    # Model confidence (1.0 - normalized_entropy)
    "time_to_first_token_ms": float,
    "total_time_ms": float,
    "prefill_tps": float,
    "decode_tps": float,
    "ram_usage_mb": float,
    "prefill_tokens": int,
    "decode_tokens": int,
    "total_tokens": int
}
"""
import ctypes
import json
import platform
from pathlib import Path

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_DIR = Path(__file__).parent.parent.parent
if platform.system() == "Darwin":
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.dylib"
else:
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.so"

_lib = None
if _LIB_PATH.exists():
    _lib = ctypes.CDLL(str(_LIB_PATH))

    _lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    _lib.cactus_init.restype = ctypes.c_void_p

    _lib.cactus_complete.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_char_p, TokenCallback, ctypes.c_void_p
    ]
    _lib.cactus_complete.restype = ctypes.c_int

    _lib.cactus_transcribe.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    ]
    _lib.cactus_transcribe.restype = ctypes.c_int

    _lib.cactus_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_bool
    ]
    _lib.cactus_embed.restype = ctypes.c_int

    _lib.cactus_image_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_image_embed.restype = ctypes.c_int

    _lib.cactus_audio_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_audio_embed.restype = ctypes.c_int

    _lib.cactus_reset.argtypes = [ctypes.c_void_p]
    _lib.cactus_reset.restype = None

    _lib.cactus_stop.argtypes = [ctypes.c_void_p]
    _lib.cactus_stop.restype = None

    _lib.cactus_destroy.argtypes = [ctypes.c_void_p]
    _lib.cactus_destroy.restype = None

    _lib.cactus_get_last_error.argtypes = []
    _lib.cactus_get_last_error.restype = ctypes.c_char_p

    _lib.cactus_set_telemetry_token.argtypes = [ctypes.c_char_p]
    _lib.cactus_set_telemetry_token.restype = None

    _lib.cactus_set_pro_key.argtypes = [ctypes.c_char_p]
    _lib.cactus_set_pro_key.restype = None

    _lib.cactus_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _lib.cactus_tokenize.restype = ctypes.c_int

    _lib.cactus_score_window.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.cactus_score_window.restype = ctypes.c_int

    _lib.cactus_rag_query.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t
    ]
    _lib.cactus_rag_query.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_init.argtypes = [ctypes.c_void_p]
    _lib.cactus_stream_transcribe_init.restype = ctypes.c_void_p

    _lib.cactus_stream_transcribe_insert.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    ]
    _lib.cactus_stream_transcribe_insert.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_process.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p
    ]
    _lib.cactus_stream_transcribe_process.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_finalize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
    ]
    _lib.cactus_stream_transcribe_finalize.restype = ctypes.c_int

    _lib.cactus_stream_transcribe_destroy.argtypes = [ctypes.c_void_p]
    _lib.cactus_stream_transcribe_destroy.restype = None


def cactus_init(model_path, corpus_dir=None, draft_model_path=None, speculation_length=0):
    """
    Initialize a model and return its handle.

    Args:
        model_path: Path to model weights directory
        corpus_dir: Optional path to RAG corpus directory for document Q&A
        draft_model_path: Optional path to draft model for speculative decoding
        speculation_length: Number of draft tokens to generate (0 uses default of 5)

    Returns:
        Model handle (opaque pointer) or None if initialization failed.
        Use cactus_get_last_error() to get error details.
    """
    return _lib.cactus_init(
        model_path.encode() if isinstance(model_path, str) else model_path,
        corpus_dir.encode() if corpus_dir else None,
        draft_model_path.encode() if draft_model_path else None,
        speculation_length
    )


def cactus_complete(
    model,
    messages,
    tools=None,
    temperature=None,
    top_p=None,
    top_k=None,
    max_tokens=None,
    stop_sequences=None,
    force_tools=False,
    tool_rag_top_k=None,
    confidence_threshold=None,
    callback=None
):
    """
    Run chat completion on a model.

    Args:
        model: Model handle from cactus_init
        messages: List of message dicts or JSON string
        tools: Optional list of tool definitions for function calling
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        max_tokens: Maximum tokens to generate
        stop_sequences: List of stop sequences
        force_tools: Constrain output to tool call format
        tool_rag_top_k: Select top-k relevant tools via Tool RAG (default: 2, 0 = disabled)
        confidence_threshold: Minimum confidence for local generation (default: 0.7, triggers cloud_handoff when below)
        callback: Streaming callback fn(token, token_id, user_data)

    Returns:
        JSON string with unified response format (all fields always present):
        {
            "success": bool,        # True if generation succeeded
            "error": str|null,      # Error message if failed, null otherwise
            "cloud_handoff": bool,  # True if model confidence too low, should defer to cloud
            "response": str|null,   # Generated text, null if cloud_handoff or error
            "function_calls": [],   # List of function calls if tools were used
            "confidence": float,    # Model confidence (1.0 - normalized_entropy)
            "time_to_first_token_ms": float,
            "total_time_ms": float,
            "prefill_tps": float,
            "decode_tps": float,
            "ram_usage_mb": float,
            "prefill_tokens": int,
            "decode_tokens": int,
            "total_tokens": int
        }

        When cloud_handoff is True, the model confidence dropped below confidence_threshold
        and recommends deferring to a cloud-based model for better results.
    """
    if isinstance(messages, list):
        messages_json = json.dumps(messages)
    else:
        messages_json = messages

    tools_json = None
    if tools is not None:
        if isinstance(tools, list):
            tools_json = json.dumps(tools)
        else:
            tools_json = tools

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if stop_sequences is not None:
        options["stop_sequences"] = stop_sequences
    if force_tools:
        options["force_tools"] = True
    if tool_rag_top_k is not None:
        options["tool_rag_top_k"] = tool_rag_top_k
    if confidence_threshold is not None:
        options["confidence_threshold"] = confidence_threshold

    options_json = json.dumps(options) if options else None

    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_complete(
        model,
        messages_json.encode() if isinstance(messages_json, str) else messages_json,
        buf, len(buf),
        options_json.encode() if options_json else None,
        tools_json.encode() if tools_json else None,
        cb, None
    )
    return buf.value.decode("utf-8", errors="ignore")


def cactus_transcribe(model, audio_path, prompt="", callback=None):
    """
    Transcribe audio using a Whisper model.

    Args:
        model: Whisper model handle from cactus_init
        audio_path: Path to audio file (WAV format)
        prompt: Whisper prompt for language/task (e.g., "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
        callback: Optional streaming callback fn(token, token_id, user_data)

    Returns:
        JSON string with response: {"success": bool, "response": str, ...}
    """
    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_transcribe(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        prompt.encode() if isinstance(prompt, str) else prompt,
        buf, len(buf),
        None, cb, None, None, 0
    )
    return buf.value.decode()


def cactus_embed(model, text, normalize=False):
    """
    Get text embeddings.

    Args:
        model: Model handle from cactus_init
        text: Text to embed
        normalize: L2-normalize embeddings (default: False)

    Returns:
        List of floats representing the embedding vector.
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_embed(
        model,
        text.encode() if isinstance(text, str) else text,
        buf, ctypes.sizeof(buf), ctypes.byref(dim), normalize
    )
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    """
    Get image embeddings from a VLM.

    Args:
        model: Model handle from cactus_init
        image_path: Path to image file

    Returns:
        List of floats representing the image embedding vector.
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_image_embed(
        model,
        image_path.encode() if isinstance(image_path, str) else image_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    """
    Get audio embeddings from a Whisper model.

    Args:
        model: Whisper model handle from cactus_init
        audio_path: Path to audio file (WAV format)

    Returns:
        List of floats representing the audio embedding vector.
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_audio_embed(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_reset(model):
    """Reset model state (clear KV cache). Call between unrelated conversations."""
    _lib.cactus_reset(model)


def cactus_stop(model):
    """Stop an ongoing generation (useful with streaming callbacks)."""
    _lib.cactus_stop(model)


def cactus_destroy(model):
    """Free model memory. Always call when done."""
    _lib.cactus_destroy(model)


def cactus_get_last_error():
    """Get the last error message, or None if no error."""
    result = _lib.cactus_get_last_error()
    return result.decode() if result else None


def cactus_set_telemetry_token(token):
    """Set telemetry token for usage tracking."""
    _lib.cactus_set_telemetry_token(
        token.encode() if isinstance(token, str) else token
    )


def cactus_set_pro_key(pro_key):
    """Set Cactus Pro key for NPU acceleration (Apple devices)."""
    _lib.cactus_set_pro_key(
        pro_key.encode() if isinstance(pro_key, str) else pro_key
    )


def cactus_tokenize(model, text: str):
    """
    Tokenize text.

    Args:
        model: Model handle from cactus_init
        text: Text to tokenize

    Returns:
        List of token IDs.
    """
    needed = ctypes.c_size_t(0)
    rc = _lib.cactus_tokenize(
        model,
        text.encode("utf-8"),
        None,
        0,
        ctypes.byref(needed),
    )
    if rc != 0:
        raise RuntimeError(f"cactus_tokenize length query failed rc={rc}")

    n = needed.value
    arr = (ctypes.c_uint32 * n)()

    rc = _lib.cactus_tokenize(
        model,
        text.encode("utf-8"),
        arr,
        n,
        ctypes.byref(needed),
    )
    if rc != 0:
        raise RuntimeError(f"cactus_tokenize fetch failed rc={rc}")

    return [arr[i] for i in range(n)]


def cactus_score_window(model, tokens, start, end, context):
    """
    Score a window of tokens for perplexity/log probability.

    Args:
        model: Model handle from cactus_init
        tokens: List of token IDs
        start: Start index of window to score
        end: End index of window to score
        context: Context size for scoring

    Returns:
        Dict with "success", "logprob", and "tokens" keys.
    """
    buf = ctypes.create_string_buffer(4096)
    n = len(tokens)
    arr = (ctypes.c_uint32 * n)(*tokens)

    _lib.cactus_score_window(
        model,
        arr,
        n,
        start,
        end,
        context,
        buf,
        len(buf),
    )
    return json.loads(buf.value.decode("utf-8", errors="ignore"))


def cactus_rag_query(model, query, top_k=5):
    """
    Query RAG corpus for relevant text chunks.

    Args:
        model: Model handle (must have been initialized with corpus_dir)
        query: Query text
        top_k: Number of chunks to retrieve (default: 5)

    Returns:
        List of dicts with "score" and "text" keys, or empty list on error.
    """
    buf = ctypes.create_string_buffer(65536)
    result = _lib.cactus_rag_query(
        model,
        query.encode() if isinstance(query, str) else query,
        buf, len(buf), top_k
    )
    if result != 0:
        return []
    return json.loads(buf.value.decode("utf-8", errors="ignore"))


def cactus_stream_transcribe_init(model):
    """
    Initialize streaming transcription session.

    Args:
        model: Whisper model handle from cactus_init

    Returns:
        Stream handle for use with other stream_transcribe functions.
    """
    return _lib.cactus_stream_transcribe_init(model)


def cactus_stream_transcribe_insert(stream, pcm_data):
    """
    Insert audio data into streaming transcription buffer.

    Args:
        stream: Stream handle from cactus_stream_transcribe_init
        pcm_data: PCM audio data as bytes or list of uint8

    Returns:
        0 on success, -1 on error.
    """
    if isinstance(pcm_data, bytes):
        arr = (ctypes.c_uint8 * len(pcm_data)).from_buffer_copy(pcm_data)
    else:
        arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
    return _lib.cactus_stream_transcribe_insert(stream, arr, len(arr))


def cactus_stream_transcribe_process(stream, options=None):
    """
    Process buffered audio and return transcription.

    Args:
        stream: Stream handle from cactus_stream_transcribe_init
        options: Optional JSON string with options (e.g., {"confirmation_threshold": 0.95})

    Returns:
        JSON string with "success", "confirmed", and "pending" keys.
    """
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_stream_transcribe_process(
        stream, buf, len(buf),
        options.encode() if options else None
    )
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_finalize(stream):
    """
    Finalize streaming transcription and get final result.

    Args:
        stream: Stream handle from cactus_stream_transcribe_init

    Returns:
        JSON string with "success" and "confirmed" keys.
    """
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_stream_transcribe_finalize(stream, buf, len(buf))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_destroy(stream):
    """Free streaming transcription resources."""
    _lib.cactus_stream_transcribe_destroy(stream)

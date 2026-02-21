#!/usr/bin/env python3
"""
Gemma 3 270M Full E2E Decode Benchmark
Compares cactus, llama.cpp, and MLX using each framework's native inference.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
WEIGHTS_DIR = PROJECT_ROOT / "weights"

DEFAULT_PREFILL = 1024
DEFAULT_TOKENS = 100
DEFAULT_RUNS = 3

GATSBY_PROMPT_100 = 'Continue the following text:\n\nIn my younger and more vulnerable years my father gave me some advice that I\'ve been turning over in my mind ever since. "Whenever you feel like criticizing anyone," he told me, "just remember that all the people in this world haven\'t had the advantages that you\'ve had." He didn\'t say any more, but we\'ve always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.'
GATSBY_PROMPT_1K = 'Continue the following text:\n\nIn my younger and more vulnerable years my father gave me some advice that I\'ve been turning over in my mind ever since. "Whenever you feel like criticizing anyone," he told me, "just remember that all the people in this world haven\'t had the advantages that you\'ve had." He didn\'t say any more, but we\'ve always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that. In consequence, I\'m inclined to reserve all judgements, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores. The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician, because I was privy to the secret griefs of wild, unknown men. Most of the confidences were unsought-frequently I have feigned sleep, preoccupation, or a hostile levity when I realized by some unmistakable sign that an intimate revelation was quivering on the horizon; for the intimate revelations of young men, or at least the terms in which they express them, are usually plagiaristic and marred by obvious suppressions. Reserving judgements is a matter of infinite hope. I am still a little afraid of missing something if I forget that, as my father snobbishly suggested, and I snobbishly repeat, a sense of the fundamental decencies is parcelled out unequally at birth. And, after boasting this way of my tolerance, I come to the admission that it has a limit. Conduct may be founded on the hard rock or the wet marshes, but after a certain point I don\'t care what it\'s founded on. When I came back from the East last autumn I felt that I wanted the world to be in uniform and at a sort of moral attention forever; I wanted no more riotous excursions with privileged glimpses into the human heart. Only Gatsby, the man who gives his name to this book, was exempt from my reaction-Gatsby, who represented everything for which I have an unaffected scorn. If personality is an unbroken series of successful gestures, then there was something gorgeous about him, some heightened sensitivity to the promises of life, as if he were related to one of those intricate machines that register earthquakes ten thousand miles away. This responsiveness had nothing to do with that flabby impressionability which is dignified under the name of the creative temperament-it was an extraordinary gift for hope, a romantic readiness such as I have never found in any other person and which it is not likely I shall ever find again. No-Gatsby turned out all right at the end; it is what preyed on Gatsby, what foul dust floated in the wake of his dreams that temporarily closed out my interest in the abortive sorrows and short-winded elations of men. My family have been prominent, well-to-do people in this Middle Western city for three generations. The Carraways are something of a clan, and we have a tradition that we\'re descended from the Dukes of Buccleuch, but the actual founder of my line was my grandfather\'s brother, who came here in fifty-one, sent a substitute to the Civil War, and started the wholesale hardware business that my father carries on today. I never saw this great-uncle, but I\'m supposed to look like him-with special reference to the rather hard-boiled painting that hangs in father\'s office. I graduated from New Haven in 1915, just a quarter of a century after my father, and a little later I participated in that delayed Teutonic migration known as the Great War. I enjoyed the counter-raid so thoroughly that I came back restless. Instead of being the warm centre of the world, the Middle West now seemed like the ragged edge of the universe-so I decided to go East and learn the bond business. Everybody I knew was in the bond business, so I supposed it could support one more single man. All my aunts and uncles talked it over as if they were choosing a prep school for me, and finally said, "Why-ye-es," with very grave, hesitant faces. Father agreed to finance me for a year, and after various delays I came East, permanently, I thought, in the spring of twenty-two. The practical thing was to find rooms in the city, but it was a warm season, and I had just left a country of wide lawns and friendly trees, so when a young man at the office suggested that we take a house together in a commuting town, it sounded like a great idea. He found the house, a weather-beaten cardboard bungalow at eighty a month, but at the last minute the firm ordered him to Washington, and I went out to the country alone. I had a dog-at least I had him for a few days until he ran away-and an old Dodge and a'


class BenchResult:
    def __init__(self, framework, model_format, prefill_tps=0, decode_tps=0,
                 ttft_ms=0, prefill_tokens=0, decode_tokens=0, error=None):
        self.framework = framework
        self.model_format = model_format
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.ttft_ms = ttft_ms
        self.prefill_tokens = prefill_tokens
        self.decode_tokens = decode_tokens
        self.error = error


def run_cactus(prompt, prefill_tokens, tokens, runs, prec="int8"):
    if prec == "int4":
        return BenchResult("cactus", "INT4", error="no INT4 model available")

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))
        import cactus
    except Exception as e:
        return BenchResult("cactus", "INT8", error=f"import failed: {e}")

    model_path = str(WEIGHTS_DIR / "gemma-3-270m-it-int8")
    model = cactus.cactus_init(model_path)
    if not model:
        return BenchResult("cactus", "INT8", error="failed to init model")

    messages = json.dumps([{"role": "user", "content": prompt}])
    options = json.dumps({"temperature": 0.0, "max_tokens": tokens})

    decode_tps_list = []
    prefill_tps_list = []
    ttft_list = []

    for r in range(runs):
        print(f"  Run {r+1}/{runs}...", flush=True)
        result = cactus.cactus_complete(model, messages, options)
        data = json.loads(result)
        if data.get("decode_tps", 0) > 0:
            decode_tps_list.append(data["decode_tps"])
            prefill_tps_list.append(data.get("prefill_tps", 0))
            ttft_list.append(data.get("time_to_first_token_ms", 0))
        cactus.cactus_reset(model)

    cactus.cactus_destroy(model)

    if not decode_tps_list:
        return BenchResult("cactus", "INT8", error="no valid results")

    return BenchResult(
        "cactus", "INT8",
        prefill_tps=sum(prefill_tps_list) / len(prefill_tps_list),
        decode_tps=sum(decode_tps_list) / len(decode_tps_list),
        ttft_ms=sum(ttft_list) / len(ttft_list),
        prefill_tokens=int(data.get("prefill_tokens", 0)),
        decode_tokens=int(data.get("decode_tokens", 0)),
    )


def run_llamacpp(prompt, prefill_tokens, tokens, runs, prec="int8"):
    llama_bench = shutil.which("llama-bench")
    if not llama_bench:
        return BenchResult("llama.cpp", "GGUF", error="llama-bench not found")

    if prec == "int4":
        gguf_path = WEIGHTS_DIR / "google_gemma-3-270m-it-Q4_0.gguf"
        fmt = "Q4_0 GGUF"
    else:
        gguf_path = WEIGHTS_DIR / "gemma-3-270m-it-Q8_0.gguf"
        fmt = "Q8_0 GGUF"
    if not gguf_path.exists():
        return BenchResult("llama.cpp", fmt, error=f"GGUF not found at {gguf_path}")

    # llama-bench reports pp (prompt processing) and tg (text generation) separately
    # -p = prompt tokens, -n = generation tokens, -r = repetitions
    cmd = [
        llama_bench,
        "-m", str(gguf_path),
        "-p", str(prefill_tokens),
        "-n", str(tokens),    # generation tokens for decode benchmark
        "-r", str(runs),
        "-o", "json",
        "-fa", "1",           # flash attention
        "-ngl", "0",          # CPU-only (no GPU offload)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    try:
        entries = json.loads(result.stdout)
        prefill_tps = 0
        decode_tps = 0
        for entry in entries:
            # n_prompt > 0 means prefill test, n_gen > 0 means decode test
            n_prompt = entry.get("n_prompt", 0)
            n_gen = entry.get("n_gen", 0)
            tps = entry.get("avg_ts", 0)

            if n_prompt > 0 and n_gen == 0:
                prefill_tps = tps
            elif n_gen > 0 and n_prompt == 0:
                decode_tps = tps

        ttft_ms = (prefill_tokens / prefill_tps * 1000) if prefill_tps > 0 else 0
        return BenchResult(
            "llama.cpp", fmt,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            ttft_ms=ttft_ms,
            prefill_tokens=prefill_tokens,
            decode_tokens=tokens,
        )
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: parse stderr for timing info
        return parse_llamacpp_stderr(result.stderr, tokens, runs)


def parse_llamacpp_stderr(stderr, tokens, runs):
    """Parse llama.cpp timing output from stderr as fallback."""
    prefill_tps = 0
    decode_tps = 0

    for line in stderr.split("\n"):
        # prompt eval time = X ms / N tokens (X ms per token, X tokens per second)
        m = re.search(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second", line)
        if m:
            prefill_tps = float(m.group(2))

        # eval time = X ms / N runs (X ms per token, X tokens per second)
        m = re.search(r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second", line)
        if m:
            decode_tps = float(m.group(2))

    if prefill_tps > 0 or decode_tps > 0:
        return BenchResult("llama.cpp", "Q8_0 GGUF", prefill_tps=prefill_tps,
                           decode_tps=decode_tps, decode_tokens=tokens)

    return BenchResult("llama.cpp", "Q8_0 GGUF", error="could not parse output")


def run_mlx(prompt, prefill_tokens, tokens, runs, prec="int8"):
    try:
        import mlx_lm
        import mlx.core as mx
    except ImportError:
        return BenchResult("MLX", "FP16", error="mlx-lm not installed")

    if prec == "int4":
        model_id = "mlx-community/gemma-3-270m-it-4bit"
        fmt = "INT4"
    elif prec == "int8":
        model_id = "mlx-community/gemma-3-270m-it-8bit"
        fmt = "INT8"
    else:
        model_id = "google/gemma-3-270m-it"
        fmt = "FP16"
    print(f"  Loading MLX model ({model_id})...", flush=True)

    try:
        model, tokenizer = mlx_lm.load(model_id)
    except Exception as e:
        return BenchResult("MLX", "FP16", error=f"load failed: {e}")

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Warmup
    print("  Warmup...", flush=True)
    _ = mlx_lm.generate(model, tokenizer, prompt=formatted, max_tokens=8, verbose=False)

    prefill_tps_list = []
    decode_tps_list = []
    ttft_list = []

    for r in range(runs):
        print(f"  Run {r+1}/{runs}...", flush=True)

        # Use stream_generate for TTFT measurement
        token_count = 0
        t_start = time.perf_counter()
        ttft = None

        for response in mlx_lm.stream_generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=tokens,
        ):
            if ttft is None:
                ttft = (time.perf_counter() - t_start) * 1000
            token_count += 1

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        if ttft is None:
            ttft = total_ms

        # Estimate: prefill ~ ttft, decode ~ rest
        decode_ms = total_ms - ttft
        decode_count = max(token_count - 1, 1)
        decode_tps = (decode_count / decode_ms * 1000) if decode_ms > 0 else 0

        # Get prompt token count
        prompt_tokens = tokenizer.encode(formatted)
        prefill_count = len(prompt_tokens) if hasattr(prompt_tokens, '__len__') else 0
        prefill_tps = (prefill_count / ttft * 1000) if ttft > 0 else 0

        prefill_tps_list.append(prefill_tps)
        decode_tps_list.append(decode_tps)
        ttft_list.append(ttft)

    avg_prefill = sum(prefill_tps_list) / len(prefill_tps_list)
    avg_decode = sum(decode_tps_list) / len(decode_tps_list)
    avg_ttft = sum(ttft_list) / len(ttft_list)

    prompt_tokens = tokenizer.encode(formatted)
    n_prompt = len(prompt_tokens) if hasattr(prompt_tokens, '__len__') else 0

    return BenchResult(
        "MLX", fmt,
        prefill_tps=avg_prefill,
        decode_tps=avg_decode,
        ttft_ms=avg_ttft,
        prefill_tokens=n_prompt,
        decode_tokens=tokens,
    )


def run_litert(prompt, prefill_tokens, tokens, runs, prec="int8"):
    fmt = "INT4 TFLite" if prec == "int4" else "INT8 TFLite"
    try:
        from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
    except ImportError as e:
        return BenchResult("LiteRT", fmt, error=f"missing dep: {e}")

    litert_dir = WEIGHTS_DIR / "litert-gemma"
    if prec == "int4":
        tflite_path = str(litert_dir / "gemma3-270m-it-q4.tflite" / "gemma3-1b_q4_block32_ekv4096.tflite")
        fmt = "INT4 TFLite"
    else:
        tflite_path = str(litert_dir / "gemma3-270m-it-q8.tflite")
        fmt = "INT8 TFLite"

    if not os.path.exists(tflite_path):
        return BenchResult("LiteRT", fmt, error=f"model not found at {tflite_path}")

    print("  Loading LiteRT CompiledModel...", flush=True)
    try:
        cm = CompiledModel.from_file(tflite_path, HardwareAccelerator.CPU)
        decode_idx = cm.get_signature_index("decode")
        # Pick best prefill signature for requested prefill_tokens
        sigs = cm.get_signature_list()
        prefill_sig = None
        prefill_len = 0
        for sig_name in sigs:
            if sig_name.startswith("prefill_"):
                n = int(sig_name.split("_")[1])
                if n >= prefill_tokens and (prefill_sig is None or n < prefill_len):
                    prefill_sig = sig_name
                    prefill_len = n
        if prefill_sig is None:
            # Fall back to largest available
            for sig_name in sigs:
                if sig_name.startswith("prefill_"):
                    n = int(sig_name.split("_")[1])
                    if n > prefill_len:
                        prefill_sig = sig_name
                        prefill_len = n
        prefill_idx = cm.get_signature_index(prefill_sig) if prefill_sig else None
    except Exception as e:
        return BenchResult("LiteRT", fmt, error=f"load failed: {e}")

    # Pre-allocate zero-copy buffers
    decode_in = cm.create_input_buffers(decode_idx)
    decode_out = cm.create_output_buffers(decode_idx)
    if prefill_idx is not None:
        prefill_in = cm.create_input_buffers(prefill_idx)
        prefill_out = cm.create_output_buffers(prefill_idx)

    # Warmup
    print("  Warmup...", flush=True)
    if prefill_idx is not None:
        cm.run_by_index(prefill_idx, prefill_in, prefill_out)
    for _ in range(10):
        cm.run_by_index(decode_idx, decode_in, decode_out)

    decode_tps_list = []
    prefill_tps_list = []

    for r in range(runs):
        print(f"  Run {r+1}/{runs}...", flush=True)

        # Prefill timing
        if prefill_idx is not None:
            t_pf0 = time.perf_counter()
            cm.run_by_index(prefill_idx, prefill_in, prefill_out)
            t_pf1 = time.perf_counter()
            pf_ms = (t_pf1 - t_pf0) * 1000
            prefill_tps_list.append(prefill_len / pf_ms * 1000)

        # Decode timing
        t0 = time.perf_counter()
        for _ in range(tokens):
            cm.run_by_index(decode_idx, decode_in, decode_out)
        t1 = time.perf_counter()
        decode_ms = (t1 - t0) * 1000
        decode_tps_list.append(tokens / decode_ms * 1000)

    avg_decode = sum(decode_tps_list) / len(decode_tps_list)
    avg_prefill = sum(prefill_tps_list) / len(prefill_tps_list) if prefill_tps_list else 0
    avg_step_ms = 1000.0 / avg_decode if avg_decode > 0 else 0

    return BenchResult(
        "LiteRT", fmt,
        prefill_tps=avg_prefill,
        decode_tps=avg_decode,
        ttft_ms=avg_step_ms,
        prefill_tokens=prefill_len,
        decode_tokens=tokens,
    )


def run_onnxrt(prompt, prefill_tokens, tokens, runs, prec="int8"):
    try:
        import onnxruntime_genai as og
    except ImportError:
        return BenchResult("ONNX Runtime", "ONNX", error="onnxruntime-genai not installed")

    if prec == "int4":
        model_path = str(WEIGHTS_DIR / "gemma-3-270m-it-onnx-int4")
        fmt = "INT4"
    else:
        model_path = str(WEIGHTS_DIR / "gemma-3-270m-it-onnx")
        fmt = "FP32"
    if not os.path.exists(os.path.join(model_path, "genai_config.json")):
        return BenchResult("ONNX Runtime", fmt, error=f"model not found at {model_path}")

    print("  Loading ONNX Runtime model...", flush=True)
    try:
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
    except Exception as e:
        return BenchResult("ONNX Runtime", fmt, error=f"load failed: {e}")

    chat_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    input_tokens = tokenizer.encode(chat_prompt)
    n_prompt = len(input_tokens) if hasattr(input_tokens, '__len__') else 0

    # Warmup
    print("  Warmup...", flush=True)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=n_prompt + 8)
    gen = og.Generator(model, params)
    gen.append_tokens(input_tokens)
    for _ in range(8):
        if gen.is_done():
            break
        gen.generate_next_token()
    del gen

    decode_tps_list = []
    ttft_list = []

    for r in range(runs):
        print(f"  Run {r+1}/{runs}...", flush=True)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=n_prompt + tokens)

        t_start = time.perf_counter()
        gen = og.Generator(model, params)
        gen.append_tokens(input_tokens)

        ttft = None
        count = 0
        while not gen.is_done():
            gen.generate_next_token()
            if ttft is None:
                ttft = (time.perf_counter() - t_start) * 1000
            count += 1
            if count >= tokens:
                break

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        del gen

        if ttft is None:
            ttft = total_ms

        decode_ms = total_ms - ttft
        decode_count = max(count - 1, 1)
        decode_tps = (decode_count / decode_ms * 1000) if decode_ms > 0 else 0

        decode_tps_list.append(decode_tps)
        ttft_list.append(ttft)

    avg_decode = sum(decode_tps_list) / len(decode_tps_list)
    avg_ttft = sum(ttft_list) / len(ttft_list)
    prefill_tps = (n_prompt / avg_ttft * 1000) if avg_ttft > 0 else 0

    return BenchResult(
        "ONNX Runtime", fmt,
        prefill_tps=prefill_tps,
        decode_tps=avg_decode,
        ttft_ms=avg_ttft,
        prefill_tokens=n_prompt,
        decode_tokens=tokens,
    )


def _print_section(title, rows):
    header = f"  {'Framework':<14} {'Format':<14} {'Prefill tok/s':>14} {'Decode tok/s':>13} {'TTFT (ms)':>10}"
    print(f"  {title}")
    print(header)
    print("  " + "-" * 12 + "   " + "-" * 12 + "   " + "-" * 12 + "   " + "-" * 11 + "   " + "-" * 8)
    sorted_rows = sorted(rows, key=lambda r: r.decode_tps if not r.error else -1, reverse=True)
    for r in sorted_rows:
        if r.error:
            print(f"  {r.framework:<14} {r.model_format:<14} {'ERROR: ' + r.error}")
            continue
        ttft_str = f"{r.ttft_ms:.1f}" if r.ttft_ms > 0 else "—"
        print(f"  {r.framework:<14} {r.model_format:<14} {r.prefill_tps:>14.1f} {r.decode_tps:>13.1f} {ttft_str:>10}")
    print()


def print_table(results):
    print()
    print("=" * 90)
    print("  Gemma 3 270M — Full E2E Decode Benchmark")
    print("=" * 90)
    print()

    # Group by precision bucket
    int8_results = [r for r in results if r.model_format and "INT4" not in r.model_format
                    and "Q4" not in r.model_format and "int4" not in r.model_format.lower()]
    int4_results = [r for r in results if r not in int8_results]

    if int8_results:
        _print_section("INT8 / FP16 / FP32", int8_results)
    if int4_results:
        _print_section("INT4", int4_results)

    print("-" * 90)
    print()


def print_json_results(results):
    out = []
    for r in results:
        entry = {
            "framework": r.framework,
            "model_format": r.model_format,
            "prefill_tps": round(r.prefill_tps, 1),
            "decode_tps": round(r.decode_tps, 1),
            "ttft_ms": round(r.ttft_ms, 1),
            "prefill_tokens": r.prefill_tokens,
            "decode_tokens": r.decode_tokens,
        }
        if r.error:
            entry["error"] = r.error
        out.append(entry)
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 270M Full E2E Benchmark")
    parser.add_argument("--prefill", type=int, default=DEFAULT_PREFILL,
                        choices=[100, 1024], help="Prefill tokens: 100 or 1024 (default: 1024)")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="Decode tokens")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Runs per backend")
    parser.add_argument("--backends", default="cactus,llama,mlx,onnxrt,litert",
                        help="Comma-separated: cactus,llama,mlx,onnxrt,litert")
    parser.add_argument("--precision", default="int8,int4",
                        help="Comma-separated: int8,int4 (default: both)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    prompt = GATSBY_PROMPT_100 if args.prefill <= 100 else GATSBY_PROMPT_1K
    prefill = 100 if args.prefill <= 100 else 1024
    backends = [b.strip() for b in args.backends.split(",")]
    precisions = [p.strip().lower() for p in args.precision.split(",")]
    results = []

    if not args.json:
        print(f"\nBenchmarking Gemma 3 270M: pp{prefill} / tg{args.tokens}, {args.runs} runs")
        print(f"Precisions: {', '.join(precisions)}")
        print(f"Prompt: \"{prompt[:80]}...\" ({prefill} tokens)")
        print()

    for prec in precisions:
        if not args.json and len(precisions) > 1:
            print(f"{'='*40} {prec.upper()} {'='*40}")
            print()

        for backend in backends:
            if not args.json:
                print(f"[{backend} {prec}]")

            if backend == "cactus":
                r = run_cactus(prompt, prefill, args.tokens, args.runs, prec)
            elif backend in ("llama", "llama.cpp", "llamacpp"):
                r = run_llamacpp(prompt, prefill, args.tokens, args.runs, prec)
            elif backend == "mlx":
                r = run_mlx(prompt, prefill, args.tokens, args.runs, prec)
            elif backend in ("onnxrt", "onnx", "onnxruntime"):
                r = run_onnxrt(prompt, prefill, args.tokens, args.runs, prec)
            elif backend == "litert":
                r = run_litert(prompt, prefill, args.tokens, args.runs, prec)
            else:
                r = BenchResult(backend, "?", error=f"unknown backend '{backend}'")

            results.append(r)

            if not args.json:
                if r.error:
                    print(f"  Error: {r.error}")
                else:
                    print(f"  Prefill: {r.prefill_tps:.1f} tok/s | Decode: {r.decode_tps:.1f} tok/s"
                          + (f" | TTFT: {r.ttft_ms:.1f}ms" if r.ttft_ms > 0 else ""))
                print()

    if args.json:
        print_json_results(results)
    else:
        print_table(results)


if __name__ == "__main__":
    main()

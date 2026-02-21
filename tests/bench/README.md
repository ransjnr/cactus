# Matmul Benchmark Suite

Compares Cactus INT8/INT4 matmul kernels against other inference frameworks on identical workloads (projections from LFM2-450M).

## Quick Start (Cactus-only, no third-party deps)

```bash
cactus build
cd tests && mkdir -p build && cd build
cmake ..
make -j test_matmul_bench
./test_matmul_bench
```

This runs the `cactus_int8` (and `cactus_int4` once it lands) backends with no external dependencies.

## Adding Third-Party Frameworks

Each framework is an opt-in CMake option. Clone/download into `third_party/` at the repo root, then enable the flag at configure time.

### GGML

```bash
git clone https://github.com/ggml-org/ggml.git third_party/ggml
cmake .. -DWITH_GGML=ON
```

Builds GGML from source. Enables `ggml` and `ggml_int8` backends (Q4_0 and INT8 matmul).

### LiteRT (TFLite)

```bash
git clone https://github.com/google-ai-edge/LiteRT.git third_party/litert
cmake .. -DWITH_LITERT=ON
```

Fetches FlatBuffers + TFLite deps on first build (requires network). Builds a minimal static lib with NEON INT8 matmul kernels and 4-bit FC kernels. Enables `litert_int8` and `litert_int4` backends.

### MLX (Apple-only)

```bash
git clone https://github.com/ml-explore/mlx.git third_party/mlx
cmake .. -DWITH_MLX=ON
```

Requires macOS with Metal. Enables the `mlx` backend.

### MLC-LLM (TVM runtime)

```bash
git clone --recursive https://github.com/mlc-ai/mlc-llm third_party/mlc
cd third_party/mlc/3rdparty/tvm && mkdir -p build && cd build
cmake .. && make -j
cd ../../../../..
cmake .. -DWITH_MLC=ON
```

Requires building the TVM runtime first. Enables the `mlc` backend.

### ONNX Runtime (prebuilt)

Download the prebuilt release for your platform from https://github.com/microsoft/onnxruntime/releases and extract into `third_party/onnxruntime/` so you have:

```
third_party/onnxruntime/
  include/
  lib/
    libonnxruntime.dylib   (macOS)
    libonnxruntime.so      (Linux)
```

```bash
cmake .. -DWITH_ONNXRT=ON
```

Enables the `onnxrt` backend.

### Executorch (XNNPACK)

```bash
cmake .. -DWITH_EXECUTORCH=ON
```

Fetches XNNPACK from GitHub at configure time (no manual clone needed). Enables the `executorch` backend.

## Enabling Multiple Frameworks

Combine flags:

```bash
cmake .. -DWITH_GGML=ON -DWITH_LITERT=ON -DWITH_MLX=ON
make -j test_matmul_bench
./test_matmul_bench
```

## CLI Options

```
./test_matmul_bench [options]

  --warmup N          Warmup iterations (default: 100)
  --iterations N      Timed iterations (default: 1000)
  --threads N         Worker threads (default: all cores)
  --batch M1,M2,...   Batch sizes to test (default: 1,13,34)
  --backends FILTER   Comma-separated backend names to run
  --mode MODE         "comparison", "stack", or "both"
  --layers N          Enable stack/layer-cycling mode with N layers
```

---

# E2E Decode Benchmark (gemma_bench.py)

Full end-to-end Gemma 3 270M decode benchmark comparing cactus against llama.cpp, MLX, ONNX Runtime, and LiteRT. Each framework runs its own native inference pipeline.

## Setup

### 1. Build cactus

```bash
source ./setup
cactus build --python
```

### 2. Install external frameworks

```bash
brew install llama.cpp          # llama-bench CLI
pip install mlx-lm              # MLX (Apple Silicon GPU)
pip install onnxruntime-genai   # ONNX Runtime
pip install ai-edge-litert      # LiteRT (XNNPACK CPU)
```

### 3. Download model weights

**Cactus** (INT8) — should already exist:
```bash
cactus download google/gemma-3-270m-it
```

**llama.cpp** (GGUF):
```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('ggml-org/gemma-3-270m-it-GGUF', 'gemma-3-270m-it-Q8_0.gguf', local_dir='weights')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/google_gemma-3-270m-it-GGUF', 'google_gemma-3-270m-it-Q4_0.gguf', local_dir='weights')"
```

**MLX** — downloads automatically from HuggingFace on first run (cached in `~/.cache/huggingface/`).

**ONNX Runtime** (genai format):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('smartvest-llc/gemma-3-270m-it-genai', local_dir='weights/gemma-3-270m-it-onnx')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('smartvest-llc/gemma-3-270m-it-genai-int4-android', local_dir='weights/gemma-3-270m-it-onnx-int4')"
```

Then fix the provider config for macOS (remove NNAPI, keep empty):
```bash
# For both weights/gemma-3-270m-it-onnx/genai_config.json and the int4 variant:
# Change "provider_options": [{"nnapi": {}}, ...] to "provider_options": []
```

**LiteRT** (INT8 — requires accepting Gemma license at `litert-community/gemma-3-270m-it`):
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('litert-community/gemma-3-270m-it', 'gemma3-270m-it-q8.litertlm', local_dir='weights/litert-gemma')"
```

Then extract the TFLite model and tokenizer from the `.litertlm` container:
```python
with open('weights/litert-gemma/gemma3-270m-it-q8.litertlm', 'rb') as f:
    f.seek(4734976); open('weights/litert-gemma/gemma3-270m-it-q8.tflite','wb').write(f.read(303996400-4734976))
    f.seek(32768); open('weights/litert-gemma/tokenizer.model','wb').write(f.read(4721840-32768))
```

**LiteRT** (INT4 — convert from HuggingFace):
```bash
pip install litert-torch
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-270m-it', local_dir='weights/gemma-3-270m-it-hf')"
python -m litert_torch.generative.examples.gemma3.convert_gemma3_to_tflite \
  --model_size=270m \
  --checkpoint_path=weights/gemma-3-270m-it-hf \
  --output_path=weights/litert-gemma/gemma3-270m-it-q4.tflite \
  --quantize=dynamic_int4_block32 \
  --prefill_seq_lens=32 \
  --kv_cache_max_len=4096
```

## Running

```bash
# Full benchmark (INT8 + INT4, all frameworks, 3 runs)
python tests/bench/gemma_bench.py

# Customize
python tests/bench/gemma_bench.py --prefill 1024 --tokens 100 --runs 5
python tests/bench/gemma_bench.py --backends cactus,llama --precision int8
python tests/bench/gemma_bench.py --json  # machine-readable output
```

## CLI Options

```
python tests/bench/gemma_bench.py [options]

  --prompt TEXT        Prompt text
  --prefill N         Prefill token count (default: 1024)
  --tokens N          Decode tokens (default: 100)
  --runs N            Runs per backend (default: 3)
  --backends LIST     cactus,llama,mlx,onnxrt,litert (default: all)
  --precision LIST    int8,int4 (default: both)
  --json              JSON output
```

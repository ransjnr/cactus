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

# DwarfStar 4 — ROCm/Strix Halo Optimized Fork

Community fork of [antirez/ds4](https://github.com/antirez/ds4) with ROCm backend optimizations for AMD Strix Halo (gfx1151) and improved MTP speculative decoding. Based on the [rocm branch](https://github.com/antirez/ds4/tree/rocm).

## Before / After

Impact of this fork's optimizations on the same hardware, same model, same prompts:

| Metric | Upstream ROCm branch | This fork | Change |
|--------|---------------------|-----------|--------|
| MTP verify (2 tokens) | 370 ms | 122 ms | **3x faster** |
| MTP generation (greedy, best) | 5.33 t/s | 10.3 t/s | **+93%** |
| MTP generation (greedy, avg) | 5.33 t/s | 9.6 t/s | **+80%** |
| Thinking + post-think output | 7.6 t/s | 10+ t/s | **+32%** |
| Baseline (no MTP) | 9.3 t/s | 9.3 t/s | no change |
| Prefill | 80 t/s | 80 t/s | no change |

Upstream ROCm branch MTP was actively harmful (slower than no MTP). This fork makes it net-positive.

## Test Environment (2026-05-13)

| Component | Version |
|-----------|---------|
| CPU / APU | AMD Ryzen AI Max+ 395 (Strix Halo) |
| GPU | Radeon 8060S Graphics (gfx1151, RDNA 3.5) |
| RAM | 128 GB LPDDR5X (256 GB/s bandwidth) |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.18.14-061814-generic |
| ROCm | 7.2 |
| HIP | 7.2.53211 |
| Model | DeepSeek-V4-Flash-IQ2XXS imatrix (~87 GB) |
| MTP model | DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32 (~3.6 GB) |

## Performance Detail

| Mode | Speed | Notes |
|------|-------|-------|
| Baseline (no MTP) | 9.3 t/s | Single-token decode |
| MTP greedy (temp=0) | **10.3 t/s** | +11% on factual/code prompts |
| Thinking phase (temp=1.0) | 9.0 t/s | Rejection sampling, ~2% MTP acceptance |
| Post-thinking text (temp=0.7) | **10+ t/s** | MTP active after `</think>` |
| Tool call generation | **10+ t/s** | Greedy MTP, forced temp=0 |
| Prefill | 80 t/s | Sustained, 2048-token chunks |
| KV cache at 64k context | 883 MB | MLA compression |

### Benchmark sweep (ds4-bench, greedy, no MTP)

| Context | Prefill (t/s) | Generation (t/s) | KV Cache |
|---------|--------------|-------------------|----------|
| 2k | 80.4 | 7.84 | 50 MB |
| 4k | 83.2 | 7.79 | 77 MB |
| 8k | 82.7 | 7.72 | 130 MB |
| 16k | 81.7 | 7.58 | 238 MB |
| 32k | 80.2 | 7.39 | 453 MB |
| 64k | 74.7 | 6.99 | 883 MB |

### MTP by prompt type (greedy, draft=2)

| Prompt type | MTP (t/s) | Baseline | Improvement |
|-------------|----------|----------|-------------|
| Factual explanation | 10.3 | 9.3 | +11% |
| Code generation | 10.1 | 9.3 | +9% |
| General knowledge | 9.8 | 9.3 | +5% |
| Creative writing | 8.5 | 9.3 | -9% |

MTP helps most on predictable content (facts, code). Creative writing has lower draft acceptance and regresses.

## Hardware

**Tested:**
- AMD Ryzen AI Max+ 395 (Strix Halo), 128 GB LPDDR5X, gfx1151
- ROCm 7.2, HIP 7.2, hipBLAS
- Ubuntu (Noble), kernel 6.18

**Requirements:**
- AMD GPU with ROCm support (gfx1151 tested, other RDNA 3.5 targets may work)
- 96+ GB system RAM (model is ~87 GB)
- ROCm 6.x or 7.x with hipcc and hipBLAS

**Not supported on this branch:**
- XNACK / HSA zero-copy (gfx1151 does not support XNACK)
- Metal or NVIDIA CUDA (use upstream main for those)

## Build

```sh
make rocm ROCM_ARCH=gfx1151
```

If hipcc fails to find C++ headers (GCC version mismatch):

```sh
sudo apt install libstdc++-15-dev   # or whichever GCC version ROCm targets
```

Produces `ds4`, `ds4-server`, and `ds4-bench` in the current directory.

## Model Setup

Download the model and optional MTP support file:

```sh
./download_model.sh q2-imatrix
./download_model.sh mtp
```

Or place your own files:
- Main model: any ds4-compatible DeepSeek V4 Flash GGUF (~87 GB for IQ2_XXS)
- MTP model: `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (~3.6 GB)

## Quick Start

**Interactive CLI:**

```sh
./ds4 -m ds4flash.gguf --cuda
```

**One-shot with MTP (greedy):**

```sh
./ds4 -m ds4flash.gguf --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 --mtp-margin 0 --temp 0 --cuda \
  -p "Write a binary search in Python"
```

**Server (recommended for daily use):**

```sh
./ds4-server \
  -m ds4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 --mtp-margin 0 \
  --cuda --ctx 262144 \
  --host 0.0.0.0 --port 8080 \
  --warm-weights \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 4096
```

API is OpenAI-compatible at `http://localhost:8080/v1`, model name `deepseek-v4-flash`.

### systemd service

```ini
[Unit]
Description=DwarfStar 4 inference server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/ds4
ExecStart=/path/to/ds4/ds4-server \
    -m /path/to/model.gguf \
    --mtp /path/to/mtp.gguf \
    --mtp-draft 2 --mtp-margin 0 \
    --cuda --ctx 262144 \
    --host 0.0.0.0 --port 8080 \
    --warm-weights \
    --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 4096
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## What This Fork Changes

### 1. HIP build fix (`ds4_cuda.cu`)

`rsqrtf()` is device-only in HIP. Replaced with `1.0f/sqrtf()` for the two host-side call sites. Without this, the ROCm build fails.

### 2. Fused matmul kernels for small batch (`ds4_cuda.cu`)

The Q8_0 and F16 matmul kernels previously launched independent thread blocks per token, loading the same weight data multiple times. New fused kernels load weights once and compute dot products for all tokens (up to 4) in a single block.

**Impact:** MTP batch verification dropped from 370ms to 122ms for 2 tokens. This is the change that made MTP net-positive on ROCm.

hipBLAS `GemmStridedBatchedEx` for the attention output projection is also slower than the native Q8_0 kernel at small batch sizes on RDNA 3.5. The cublas threshold for this path was raised from 2 to 5 tokens.

These changes are ROCm/RDNA 3.5 specific. On NVIDIA hardware, cuBLAS may be faster for small batches — do not apply these thresholds without testing.

### 3. Prefix-K speculative commit (`ds4.c`)

Upstream ds4 captures compressor state only after the first draft token (prefix-1), supporting cheap partial accepts for `--mtp-draft 2` only. Deeper drafts (3+) fall back to expensive full-model replay on partial accept (~111ms).

This fork generalizes to prefix-K: compressor state is captured after each draft token during verification. Partial accepts at any depth cost ~0.4ms (GPU tensor copies) instead of a full replay. Memory cost is ~8 MB per prefix position, capped at 7 positions (~56 MB total).

This change is backend-agnostic and benefits Metal, CUDA, and ROCm equally. A [standalone PR](https://github.com/antirez/ds4/pull/XXX) targeting upstream main is available.

### 4. Sampling-aware speculative decoding (`ds4.c`, `ds4_server.c`)

Upstream MTP only works with greedy decoding (temperature=0). This fork adds `ds4_session_eval_speculative_sampling` which uses rejection sampling (`min(1, p_target/p_draft)`) to allow MTP with temperature > 0.

At temperature 1.0 (thinking mode), acceptance is ~2% — roughly neutral. At lower temperatures (0.3-0.7), acceptance improves and MTP provides measurable speedup.

### 5. Post-thinking MTP (`ds4_server.c`)

Upstream forces temperature=1.0 for the entire response when thinking mode is enabled — including the visible text after `</think>`. This disables MTP on the user-visible output.

This fork narrows the override: temperature=1.0 is forced only while inside `<think>...</think>`. After `</think>`, the client's requested temperature applies, enabling MTP on the actual response. At the client's temperature of 0.7, post-thinking text runs at 10+ t/s instead of 7.6.

### 6. Experimental: n-gram lookahead draft (`ds4.c`)

Search token history for 3-gram suffix matches and propose continuations as draft tokens at zero GPU cost. Gated behind `DS4_NGRAM_DRAFT=1` environment variable.

Not currently profitable on MoE models — verification cost scales with the number of MoE experts activated across draft tokens, overwhelming the free draft generation. May help on dense models or with future verification improvements.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `DS4_MTP_TIMING=1` | Log MTP draft/verify/accept timing per round |
| `DS4_MTP_CONF_LOG=1` | Log MTP confidence and acceptance details |
| `DS4_MTP_SPEC_LOG=1` | Log first-draft misses |
| `DS4_VERIFY_PROFILE=1` | Log verify encode/gpu_sync/output_head split |
| `DS4_NGRAM_DRAFT=1` | Enable experimental n-gram lookahead drafting |
| `DS4_MTP_SPEC_DISABLE=1` | Disable MTP speculative decoding |
| `DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A=1` | Force native Q8 kernel for attention output (default threshold handles this) |
| `DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` | Log F16 weight cache allocations |

## Optimal Settings

For Strix Halo with 128 GB RAM:

- `--mtp-draft 2` — optimal for current MTP model quality (single transformer layer)
- `--mtp-margin 0` — always attempt verification (our fused kernels make it cheap)
- `--ctx 262144` — 256k context, uses ~4.8 GB for buffers, fits comfortably
- `--warm-weights` — eliminates first-request page faults
- `--kv-disk-dir` — enables KV cache persistence for agent workflows

Draft depths >2 are not profitable with the current MTP model but the prefix-K infrastructure supports up to 8 for future models.

## Key Findings

- **Memory bandwidth (256 GB/s) is the hard ceiling** for generation speed on this hardware. No software optimization can exceed it.
- **hipBLAS is slower than custom kernels** for very small batch sizes (n_tok <= 4) on RDNA 3.5. cuBLAS on NVIDIA may behave differently.
- **MoE expert divergence** adds ~33% overhead per additional verified token. Each draft token may route to different experts, requiring extra weight reads. This is fundamental to MoE and limits speculative decoding gains compared to dense models.
- **XNACK is not supported** on gfx1151. HSA zero-copy via raw mmap pointers is not possible; `hipHostRegister` is required.
- **ROCm tuning env vars** (`GPU_MAX_HW_QUEUES`, `HSA_ENABLE_SDMA`, `HIP_FORCE_DEV_KERNARG`) had no measurable effect on this hardware.
- **Context size has no impact on generation speed** — MLA compressed KV cache is so small that even 256k context doesn't affect per-token speed.

## Upstream

This fork is based on the [rocm branch](https://github.com/antirez/ds4/tree/rocm) of [antirez/ds4](https://github.com/antirez/ds4). The upstream project supports Metal (primary), CUDA, and CPU backends. See the upstream README for general ds4 documentation including model weights, CLI usage, server API, disk KV cache, thinking modes, tool calling, agent client setup, and test vectors.

## Acknowledgements

To [antirez](https://github.com/antirez) for building ds4 and making DeepSeek V4 Flash feel like a finished local inference experience. To the ds4 community for maintaining the ROCm branch. And to [llama.cpp](https://github.com/ggml-org/llama.cpp) and GGML for the ecosystem that made all of this possible.

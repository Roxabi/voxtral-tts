# Voxtral-4B-TTS: Fast int4 Quantized Inference

**57-62 fps | 3.7 GB VRAM | Near-lossless quality | RTX 3090**

int4 quantized inference for Mistral's [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) text-to-speech model. Achieves **4.6x real-time** speech generation with **54% VRAM reduction** using [torchao](https://github.com/pytorch/ao) int4 quantization with the [HQQ](https://github.com/mobiusml/hqq) algorithm.

## Results

| Metric | BF16 (original) | int4 HQQ (ours) | Change |
|--------|:---------------:|:----------------:|:------:|
| **Inference Speed** | 42 fps | **59 fps** | +40% |
| **VRAM** | 8.0 GB | **3.7 GB** | -54% |
| **Real-Time Factor** | 0.30 | **0.21** | 4.8x real-time |
| **Audio Quality** | Baseline | Near-lossless | Whisper transcription match |
| **Load Time** | ~15s | ~21s | +6s (one-time quantization) |

> Benchmarked on RTX 3090 (24 GB, SM86 Ampere), CUDA 12.x, PyTorch 2.11+, flow_steps=3, cfg_alpha=1.0

### Speed Breakdown by Configuration

| Configuration | FPS | RTF | VRAM | Notes |
|---------------|:---:|:---:|:----:|-------|
| BF16 baseline | 42 | 0.30 | 8.0 GB | Original model, no optimization |
| int4 HQQ backbone | 46 | 0.27 | 3.7 GB | Weight quantization only |
| + torch.compile acoustic | 50 | 0.25 | 3.7 GB | Compiled flow-matching decoder |
| **+ static KV cache + compile all** | **58** | **0.22** | **3.7 GB** | **Full optimization stack** |
| int4 + KV cache quant (Hadamard) | 36 | 0.35 | 3.7 GB | Slower -- rotation overhead dominates |

### Flow Steps vs Speed vs Quality

| Flow Steps | CFG | FPS | RTF | Whisper Quality |
|:----------:|:---:|:---:|:---:|:---------------:|
| 8 | 1.2 | 19 | 0.64 | Best |
| 8 | off | 31 | 0.40 | Good |
| 4 | off | 48 | 0.26 | Good |
| **3** | **off** | **55** | **0.23** | **Good (sweet spot)** |
| 2 | off | 66 | 0.19 | Degraded (repetitions) |

### Fresh Benchmark (int4 HQQ, flow_steps=3, cfg_alpha=1.0)

```
[int4] VRAM: 3.68 GB | Quantized in 7.3s

--- Speed ---
  [61 fps] "The quick brown fox jumps over the lazy dog."
  [56 fps] "Hello world, how are you today?"
  [57 fps] "Paris is a beautiful city with many famous landmarks."
  [62 fps] "Technology advances rapidly in the modern world."

--- Whisper Transcription ---
  In:  "The quick brown fox jumps over the lazy dog."
  Out: "The quick brown flocks jumps over the lazy dog."

  In:  "Hello world, how are you today?"
  Out: "Hello world, how are you today?"              ✓ exact

  In:  "Paris is a beautiful city with many famous landmarks."
  Out: "Harris is a beautiful city with many famous landmarks."

  In:  "Technology advances rapidly in the modern world."
  Out: "Technology advances rapidly in the modern world."   ✓ exact
```

---

## Comparison with Other Implementations

| | **This repo** | [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) | [voxtral-tts.c](https://github.com/mudler/voxtral-tts.c) |
|---|:---:|:---:|:---:|
| **Language** | Python/PyTorch | Rust/Burn/WGPU | Pure C |
| **Quantization** | int4 HQQ (optimal) | Q4_0 GGUF (RTN) | None (BF16) |
| **Speed** | **RTF 0.21** | RTF 0.97 | RTF 7.3 |
| **Real-time?** | **4.8x faster** | ~1x (barely) | 7x slower |
| **VRAM** | 3.7 GB | 2.67 GB | ~8 GB |
| **Quality** | Near-lossless (HQQ) | 8.49% WER (Q4 RTN) | No metrics |
| **Platform** | CUDA (NVIDIA) | Cross-platform + Browser | CPU + optional CUDA |
| **Tested GPU** | RTX 3090 | Unspecified | DGX Spark (Blackwell) |

**Key takeaway:** This repo is **~5x faster** than the Rust implementation and **~35x faster** than the C implementation (despite the C version running on newer Blackwell hardware). The HQQ algorithm is critical -- simple round-to-nearest (Q4_0) degrades quality significantly.

---

## Quick Start

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USER/voxtral-int4.git
cd voxtral-int4

# Create venv
python3 -m venv venv
source venv/bin/activate
pip install torch torchao hqq safetensors soundfile numpy

# Download model (~7.5 GB)
pip install huggingface_hub
huggingface-cli download mistralai/Voxtral-4B-TTS-2603 --local-dir models/original
```

### Run

```bash
# Generate speech
./run.sh "Hello, how are you today?"

# Custom output path
./run.sh "Your text here" output.wav

# Run benchmark
./run.sh
```

### Python API

```python
import torch
from torchao_inference import load_model_int4
from generate_fast import TekkenTokenizer, generate_speech_fast

MODEL_DIR = "models/original"
model = load_model_int4(MODEL_DIR, device="cuda")
tok = TekkenTokenizer(f"{MODEL_DIR}/tekken.json")

with torch.inference_mode():
    audio, gen_time = generate_speech_fast(
        model, tok,
        "Hello world, how are you today?",
        voice_name="neutral_female",
        voice_dir=f"{MODEL_DIR}/voice_embedding",
        max_frames=300, device="cuda",
        flow_steps=3, cfg_alpha=1.0,
    )

# audio is a numpy array at 24kHz
import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### Available Voices

20 voices across 9 languages:

| Voice | Languages |
|-------|-----------|
| `neutral_female`, `neutral_male` | English |
| `cheerful_female`, `cheerful_male` | English |
| `fr_female`, `fr_male` | French |
| `de_female`, `de_male` | German |
| `es_female`, `es_male` | Spanish |
| `it_female`, `it_male` | Italian |
| `pt_female`, `pt_male` | Portuguese |
| `nl_female`, `nl_male` | Dutch |
| `hi_female`, `hi_male` | Hindi |

---

## How It Works

### Architecture

Voxtral-4B-TTS is a three-stage model:

```
Text → [LLM Backbone] → hidden states → [Acoustic Transformer] → mel frames → [Codec Decoder] → 24kHz audio
         3.03B params        394M params (flow-matching)       152M params
         26 layers           3 layers, 8 Euler steps           4-stage conv
         GQA (32Q/8KV)       CFG guidance (alpha=1.2)          ALiBi attention
```

### Our Optimization

We quantize **only the LLM backbone** (77% of parameters) to int4, keeping the acoustic transformer and codec decoder at full BF16 precision:

```
Component          | Original | Quantized | Strategy
-------------------|----------|-----------|------------------
LLM Backbone       | BF16     | int4 HQQ  | 77% of params, tolerates quantization
Acoustic Transformer| BF16    | BF16      | Stochastic flow-matching, needs precision
Codec Decoder      | BF16     | BF16      | Audio-critical convolutions
Embeddings         | BF16     | BF16      | Tied output projection
```

### Why HQQ, Not Round-to-Nearest?

Standard int4 quantization (RTN with min-max scaling) **produces garbage audio** -- the model can't predict end-of-audio tokens and generates gibberish indefinitely. HQQ uses iterative half-quadratic optimization to find optimal scale and zero-point parameters, preserving the precision needed for TTS.

We tested this directly:
- **int4 RTN:** 66 fps, but Whisper transcription completely wrong, infinite generation
- **int4 HQQ:** 59 fps, near-perfect Whisper transcription

### Key Technical Details

| Choice | Why |
|--------|-----|
| **HQQ algorithm** | Minimizes quantization error iteratively (not naive min-max) |
| **tinygemm kernel** | PyTorch built-in CUDA kernel, fuses dequant+matmul in 1 launch per layer |
| **TILE_PACKED_TO_4D** | Required packing format for SM86 (RTX 3090). Default torchao format needs SM90+ (H100) |
| **torch.inference_mode()** | +7 fps over torch.no_grad() |
| **3 flow steps** | 2.7x faster acoustic decoder vs default 8 steps, minimal quality loss |
| **Static KV cache** | Pre-allocated buffers, CUDA graph compatible |
| **Selective quantization** | Only backbone quantized; acoustic + codec stay BF16 |

---

## What We Tried (8 Approaches)

This project explored 8 different quantization approaches before finding the winning solution. Here's the summary:

| # | Approach | FPS | VRAM | Quality | Verdict |
|:-:|----------|:---:|:----:|:-------:|---------|
| 1 | TurboQuant native (on-the-fly dequant) | 2 | 5.2 GB | Good | **FAILED** -- 4,400 kernel launches/token from per-group rotation |
| 2 | TurboQuant dequant-to-BF16 at load | 31 | 8.0 GB | Good | Works but no VRAM savings (disk only) |
| 3 | LazyDequantLinear (streaming buffer) | 4-7 | 5.2 GB | Good | **FAILED** -- dequant (7ms) slower than matmul (0.078ms) |
| 4 | CPU offload via PCIe | 4 | Low | Good | **FAILED** -- PCIe bandwidth bottleneck |
| 5 | HQQ + GemLite Triton kernels | 41 | 3.7 GB | Good | Partial -- Python dispatch overhead across 182 layers |
| 6 | Fused Triton for TurboQuant | N/A | N/A | N/A | Research showed max 8-14 fps (rotation FLOPs dominate) |
| 7 | torchao int4 + RTN | 66 | 3.7 GB | **Garbage** | **FAILED** -- wrong tokens, no end-of-audio detection |
| **8** | **torchao int4 + HQQ** | **59** | **3.7 GB** | **Near-perfect** | **THE SOLUTION** |

### Key Lessons

1. **Quantization algorithm matters more than kernel speed** -- HQQ vs RTN is the difference between working and broken
2. **Kernel launch overhead kills throughput** -- 4,400 launches (TurboQuant) = 2 fps vs 182 launches (tinygemm) = 59 fps
3. **Python dispatch is real overhead** -- GemLite's 0.072ms kernel still loses to tinygemm's 0.078ms kernel because of 7ms Python overhead across 182 layers
4. **KV cache quantization is irrelevant for short-sequence TTS** -- at batch=1 seq=200, KV cache is only 1.3% of bandwidth
5. **Packing format is GPU-dependent** -- SM86 (RTX 3090) needs TILE_PACKED_TO_4D; default format silently fails

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the complete research record including KV cache quantization analysis, quality evaluations, and detailed breakdowns.

---

## Project Structure

```
src/
  torchao_inference.py   # THE SOLUTION: int4 HQQ quantization + tinygemm inference
  generate_fast.py       # Optimized TTS: static cache, 3-step flow, torch.compile
  model.py               # Full Voxtral-4B-TTS architecture (backbone + acoustic + codec)
  generate.py            # Original TTS generation pipeline
  load_model.py          # Weight loading and key mapping
  weight_utils.py        # Weight separation (backbone vs acoustic vs codec)
  benchmark_all.py       # End-to-end benchmark suite (5 configs, Whisper evaluation)

run.sh                   # Easy entry point for generation and benchmarking
RESEARCH_LOG.md          # Complete research log: 8 approaches, benchmarks, lessons learned
```

## Requirements

- **GPU:** NVIDIA with compute capability >= 8.0 (RTX 3090, A100, RTX 4090, H100, etc.)
- **VRAM:** 4 GB minimum (3.7 GB model + working memory)
- **Python:** 3.10+
- **PyTorch:** 2.11+ with CUDA
- **Key packages:** `torchao>=0.16`, `hqq`, `safetensors`, `soundfile`, `numpy`
- **Optional:** `whisper` (for quality evaluation)

## Model

This repo uses Mistral's [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) (3.5B params, 7.5 GB BF16). The model is downloaded from HuggingFace and quantized at load time -- no pre-quantized checkpoints needed.

## License

Code in this repo is MIT. Model weights are subject to [Mistral's license](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for the Voxtral-4B-TTS model
- [torchao](https://github.com/pytorch/ao) for int4 quantization infrastructure
- [HQQ](https://github.com/mobiusml/hqq) for the half-quadratic quantization algorithm
- [TrevorS/voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) -- Rust/WGPU implementation with browser support
- [mudler/voxtral-tts.c](https://github.com/mudler/voxtral-tts.c) -- Pure C reference implementation

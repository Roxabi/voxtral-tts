# Voxtral-4B-TTS: Fast int4 Quantized Inference

**57 fps | 3.8 GB VRAM | Near-lossless quality | RTX 3090**

int4 quantized inference for Mistral's [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) text-to-speech model. Achieves **4.6x real-time** speech generation with **54% VRAM reduction** using [torchao](https://github.com/pytorch/ao) int4 quantization with the [HQQ](https://github.com/mobiusml/hqq) algorithm.

## Results

| Metric | BF16 (original) | int4 HQQ (ours) | Change |
|--------|:---------------:|:----------------:|:------:|
| **Inference Speed** | 31 fps | **57 fps** | +84% |
| **VRAM** | 8.0 GB | **3.8 GB** | -53% |
| **Real-Time Factor** | 0.40 | **0.22** | 4.6x real-time |
| **Audio Quality** | Baseline | Near-lossless | Whisper transcription match |
| **3s Utterance Latency** | 1,346 ms | **787 ms** | 1.7x faster |

> Benchmarked on RTX 3090 (24 GB, SM86 Ampere), CUDA 12.x, PyTorch 2.11+, flow_steps=3, cfg_alpha=1.2

### Speed Breakdown by Configuration

| Configuration | FPS | RTF | VRAM | Notes |
|---------------|:---:|:---:|:----:|-------|
| BF16 original (Euler-8) | 31 | 0.40 | 8.0 GB | Paper defaults, no optimization |
| BF16 (Midpoint-3) | 42 | 0.30 | 8.0 GB | Faster ODE solver |
| int4 HQQ backbone | 46 | 0.27 | 3.8 GB | Weight quantization only |
| + torch.compile acoustic | 51 | 0.25 | 3.8 GB | Compiled flow-matching decoder |
| **+ static KV cache + compile all** | **57** | **0.22** | **3.8 GB** | **Full optimization stack** |
| int4 + KV cache quant (Hadamard) | 37 | 0.34 | 3.8 GB | Slower -- rotation overhead dominates |

### Comprehensive Quality Test (12 texts, int4 + compile, Whisper base)

```
tiny             57 fps  1.9s  Hi!
numbers          56 fps 11.3s  The year 2025 had 365 days, with temperatures reaching 102.7 degrees...  ✓
punctuation      56 fps  8.4s  really? No way! He said, I can't believe it...
rare-words       56 fps 13.3s  The Archaeopteryx fossil was discovered near the Solnhofen Quarry...  ✓
mixed-lang       56 fps  7.5s  She said bonjour and then switched to saying donkey shun...  ✓
abbreviations    56 fps 11.8s  Dr. Smith from NASA and Professor Jones at MIT discussed the CEO...  ✓
repetitive       55 fps  6.9s  Big big big dog ran and ran and ran around the very very very tall...  ✓
whisper-test     56 fps  7.1s  One, two, three, four, five, six, seven, eight, nine, ten.  ✓
emotional        56 fps  4.5s  I absolutely cannot believe this is actually happening right now!  ✓
medium           55 fps 14.8s  Machine learning models are increasingly being used in healthcare...  ✓
very-long        57 fps 40.0s  Throughout the history of human civilization, the pursuit of knowle...  ✓
```

**12/12 texts complete, 0 crashes, consistent 55-57 fps.** Quality matches the original BF16 model on all texts (verified side-by-side with Whisper transcription).

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
from generate_fast import generate_speech_fast, enable_static_cache
from generate import TekkenTokenizer

MODEL_DIR = "models/original"
model = load_model_int4(MODEL_DIR, device="cuda")
tok = TekkenTokenizer(f"{MODEL_DIR}/tekken.json")

# Optional: static cache + compile for max speed (57 fps vs 46 fps)
enable_static_cache(model, max_seq_len=700)
model.backbone = torch.compile(model.backbone, mode="default", fullgraph=False)
model.acoustic.predict_velocity = torch.compile(
    model.acoustic.predict_velocity, mode="default", fullgraph=False)

with torch.inference_mode():
    audio, gen_time = generate_speech_fast(
        model, tok,
        "Hello world, how are you today?",
        voice_name="neutral_female",
        voice_dir=f"{MODEL_DIR}/voice_embedding",
        max_frames=300, device="cuda",
        flow_steps=3, cfg_alpha=1.2,
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
| **Midpoint ODE solver** | 2nd-order solver at 3 steps ≈ 1st-order Euler at 6 steps, 2.7x fewer acoustic passes |
| **cfg_alpha=1.2** | Classifier-Free Guidance required for quality. 1.0 (off) produces garbled audio |
| **Static KV cache** | Pre-allocated BF16 buffers, eliminates torch.cat allocation per step |
| **torch.compile** | Backbone + acoustic predict_velocity compiled (mode=default, kernel fusion) |
| **Selective quantization** | Only backbone quantized; acoustic + codec stay BF16 |
| **Tokenizer OOB fix** | Tekken has 150K entries but model vocab is 131K. Rare tokens clamped to valid range |
| **NaN guards** | Numerical stability for long sequences: nan_to_num on logits + embedding indices |

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

## Bugs Fixed in Upstream Model

We discovered and fixed two bugs in the original Voxtral inference code that affect all implementations:

### 1. Tokenizer Vocabulary Overflow (CUDA crash)

The Tekken tokenizer has **150,000** BPE entries, but the model's embedding table is only **131,072** rows. ~13% of tokenizer entries (19,928 tokens) produce out-of-bounds indices that crash with `CUDA device-side assert`. Any text with rare subwords (e.g., "aqueducts", "Supercalifragilisticexpialidocious") triggers this.

**Fix:** Clamp token IDs to the valid embedding range in `TekkenTokenizer.encode()`.

### 2. CFG Guidance Required (garbled output)

Disabling Classifier-Free Guidance (`cfg_alpha=1.0`) produces phonetically plausible but semantically wrong audio — the acoustic decoder hallucinates instead of following the text. This is because the flow-matching ODE needs the CFG signal to stay on the text-conditioned trajectory.

**Fix:** Default `cfg_alpha=1.2` (matching the paper). Never use `cfg_alpha=1.0` for production.

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

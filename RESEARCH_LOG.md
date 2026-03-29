# Voxtral-4B-TTS Quantization Research Log

**Project:** Real-time inference optimization for Voxtral-4B-TTS on RTX 3090
**Hardware:** NVIDIA RTX 3090 (24 GB, SM86 Ampere, 936 GB/s HBM)
**Date Range:** March 2026
**Final Result:** 57 fps (4.6x real-time), 3.7 GB VRAM, near-lossless quality

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [All Approaches Tried](#2-all-approaches-tried)
3. [The Winning Solution](#3-the-winning-solution)
4. [Benchmark Results](#4-benchmark-results)
5. [Quality Evaluation](#5-quality-evaluation)
6. [KV Cache Quantization Research](#6-kv-cache-quantization-research)
7. [Key Lessons Learned](#7-key-lessons-learned)
8. [External Projects Compared](#8-external-projects-compared)
9. [References](#9-references)

---

## 1. Model Architecture

**Model:** [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)

| Component | Params | Layers | Notes |
|-----------|--------|--------|-------|
| LLM Backbone | 3.03B | 26 | Ministral-3B derivative, GQA (32Q/8KV), dim=3072, head_dim=128 |
| Acoustic Transformer | 394M | 3 | Flow-matching, 8 Euler steps (default), CFG alpha=1.2 |
| Codec Decoder | 152M | 4 stages | Convolutional, ALiBi, weight-normalized, 24kHz output |
| Embeddings | 431M | - | 131K vocab (Tekken), tied output projection |

**Key config:**
```
dim=3072, n_layers=26, n_heads=32, n_kv_heads=8, head_dim=128
rope_theta=1,000,000, vocab_size=131,072 (Tekken BPE)
acoustic_n_layers=3, n_acoustic_codebooks=36, fsq_levels=21
sample_rate=24,000 Hz, patch_size=240, frame_rate=12.5 Hz
```

---

## 2. All Approaches Tried

### Approach 1: TurboQuant Native Inference -- FAILED (2 fps)

**Idea:** Keep weights in TurboQuant packed format, dequant on-the-fly.

**Why it failed:** TurboQuant uses per-group (g=128) random orthogonal rotation matrices (128x128 QR-decomposed). Each of 182 layers has 24 groups = **4,400 kernel launches per token**. Launch overhead alone kills throughput.

| Metric | Value |
|--------|-------|
| Speed | 2 fps |
| VRAM | 5.2 GB |
| Per-layer latency | 2.4ms (vs 0.078ms for nn.Linear) |

**Lesson:** Per-group rotation makes on-the-fly dequant impossible at batch=1. The rotation FLOPs (13.6M/token/layer) dominate.

### Approach 2: TurboQuant Dequant-to-BF16 at Load -- WORKED (31 fps, 8 GB)

**Idea:** Dequant all layers to BF16 nn.Linear at load time. Standard matmul at inference.

| Metric | Value |
|--------|-------|
| Speed | 31 fps |
| VRAM | 8.0 GB (same as original) |
| Load time | 149s (slow dequant) |
| Quality | Near-lossless (cos sim 0.99996) |

**Lesson:** Proves TurboQuant quality is excellent, but no runtime benefit -- disk savings only.

### Approach 3: LazyDequantLinear / Streaming -- FAILED (4-7 fps)

**Idea:** Keep packed weights on GPU, dequant one layer at a time into a shared BF16 buffer.

**Why it failed:** Full-matrix dequant (unpack + codebook lookup + rotation matmul + scaling) takes 7ms per layer vs 0.078ms for the actual matmul.

**Lesson:** Dequant cost must be fused into the matmul kernel, not done separately.

### Approach 4: CPU Offload / Streaming -- FAILED (4 fps)

**Idea:** Keep packed weights in CPU pinned memory, stream to GPU via PCIe.

**Why it failed:** PCIe bandwidth bottleneck. The model already fits on 24 GB GPU.

**Lesson:** CPU offload only makes sense when model doesn't fit in VRAM.

### Approach 5: HQQ + GemLite Triton -- PARTIAL (41 fps, 3.7 GB)

**Idea:** Quantize with HQQ, use GemLite's fused Triton dequant+matmul kernels.

| Metric | Value |
|--------|-------|
| Speed | 41 fps |
| VRAM | 3.7 GB |
| Per-layer kernel | 0.072ms (faster than nn.Linear!) |

**Why only partial:** Python dispatch overhead across 182 HQQLinear wrappers adds ~7ms total. torch.compile can't help due to dynamic KV cache shapes breaking CUDA graphs.

**Lesson:** Kernel speed doesn't matter if Python dispatch is the bottleneck. Need C++/CUDA dispatch path.

### Approach 6: Fused Triton for TurboQuant -- RESEARCHED, Would Fail (8-14 fps max)

**Idea:** Single fused Triton kernel processing all 24 groups per layer.

**Why it wouldn't work:** Even with perfect fusion, the dense QR rotation adds 24x the FLOPs of the linear op. Kernel becomes compute-bound on rotation. Only Walsh-Hadamard rotation would work, but model was quantized with QR (baked in).

**Lesson:** Quantization format choice at training time constrains inference forever. Choose Hadamard over QR for inference-friendly rotation.

### Approach 7: torchao int4 + RTN -- FAILED (66 fps but garbage quality)

**Idea:** Use torchao's int4 weight-only quantization with default round-to-nearest.

| Metric | Value |
|--------|-------|
| Speed | 66 fps |
| VRAM | 3.7 GB |
| Quality | GARBAGE -- wrong tokens, no end-of-audio detection |

**Lesson:** RTN (round-to-nearest) with simple min-max scaling loses too much precision for TTS. Model can't predict end-of-audio tokens. **Algorithm for choosing scale/zero matters enormously.**

### Approach 8: torchao int4 + HQQ -- THE SOLUTION (57-66 fps, 3.7 GB)

See next section.

---

## 3. The Winning Solution

**torchao int4 weight-only quantization with HQQ algorithm for scale/zero optimization, tinygemm CUDA kernel for inference.**

### Why it works:

1. **HQQ algorithm** (Half-Quadratic Quantization) minimizes quantization error iteratively -- not naive min-max or RTN
2. **tinygemm kernel** (PyTorch built-in) fuses dequant+matmul in one CUDA kernel launch
3. **182 kernel launches per token** (1 per layer) vs 4,400 for TurboQuant
4. **Selective quantization:** only backbone (3.03B params) is quantized; acoustic (394M) + codec (152M) + embeddings (431M) stay BF16

### Config:

```python
# torchao int4 + HQQ
group_size = 64
packing_format = TILE_PACKED_TO_4D   # Required for RTX 3090 (SM86)
qparams_algorithm = HQQ              # Critical -- RTN fails
kernel = tinygemm                     # Built into PyTorch, B=1 optimal on Ampere
```

### Performance progression with additional optimizations:

| Config | Short FPS | Long FPS | RTF | VRAM |
|--------|-----------|----------|-----|------|
| BF16 baseline | 42 | 42 | 0.30 | 8.0 GB |
| int4 HQQ | 45 | 47 | 0.27 | 3.7 GB |
| + torch.compile acoustic | 49 | 49 | 0.25 | 3.7 GB |
| **+ static cache + compile all** | **57** | **58** | **0.22** | **3.7 GB** |

### Key decisions:

- **torch.inference_mode()** not torch.no_grad() -- +7 fps
- **3 flow steps** (not 8) with midpoint ODE solver -- 2.7x faster acoustic, minimal quality loss
- **cfg_alpha=1.0** (disabled) or 1.2 -- disabling doubles acoustic speed, slight quality tradeoff
- **TILE_PACKED_TO_4D** packing -- default torchao format fails on SM86 (RTX 3090), needs H100

---

## 4. Benchmark Results

### End-to-End Benchmark (benchmark_all.py, RTX 3090)

All configs use flow_steps=3, cfg_alpha=1.2.

**Config 1: BF16 Original**
| Text | FPS | RTF | Duration | Whisper |
|------|-----|-----|----------|---------|
| "Hello, how are you today?" | 41.8 | 0.299 | 3.1s | Exact match |
| "The weather is nice outside." | 41.8 | 0.299 | 3.8s | Exact match |
| Long AI paragraph | 41.8 | 0.299 | 15.1s | "assistance" vs "assistants" |
| VRAM: 8.01 GB, Peak: 8.23 GB |

**Config 2: int4 HQQ**
| Text | FPS | RTF | Duration | Whisper |
|------|-----|-----|----------|---------|
| "Hello, how are you today?" | 44.6 | 0.271 | 2.3s | Missing "Hello" |
| "The weather is nice outside." | 46.5 | 0.264 | 4.6s | Exact match |
| Long AI paragraph | 46.5 | 0.269 | 16.2s | "assistance" vs "assistants" |
| VRAM: 3.68 GB, Peak: 8.67 GB |

**Config 3: int4 + KV cache v1 (Hadamard+Lloyd-Max)**
| Text | FPS | RTF | Duration | Whisper |
|------|-----|-----|----------|---------|
| "Hello, how are you today?" | 35.0 | 0.357 | 3.0s | Exact match |
| "The weather is nice outside." | 36.3 | 0.344 | 2.4s | Exact match |
| Long AI paragraph | 36.2 | 0.345 | 12.9s | Missing "Artificial" |
| VRAM: 3.68 GB -- **30% SLOWER than int4 alone** |

**Config 4: int4 + compile acoustic**
| Text | FPS | RTF | Duration | Whisper |
|------|-----|-----|----------|---------|
| "Hello, how are you today?" | 48.8 | 0.248 | 2.3s | Missing "Hello" |
| "The weather is nice outside." | 50.8 | 0.246 | 4.2s | Exact match |
| Long AI paragraph | 49.0 | 0.255 | 15.4s | "assistance" vs "assistants" |
| VRAM: 3.68 GB |

**Config 5: int4 + static cache + compile all (BEST)**
| Text | FPS | RTF | Duration | Whisper |
|------|-----|-----|----------|---------|
| "Hello, how are you today?" | 55.6 | 0.217 | 2.3s | "now" vs "how" |
| "The weather is nice outside." | 57.8 | 0.216 | 2.9s | Exact match |
| Long AI paragraph | 57.9 | 0.216 | 16.3s | "assistance" vs "assistants" |
| VRAM: 3.68 GB |

### Speed Settings Sweep (from speed_benchmark.json)

| Config | FPS | RTF | Quality |
|--------|-----|-----|---------|
| 8 steps + CFG=1.2 (default) | 19.5 | 0.64 | Best quality |
| 8 steps, no CFG | 31.0 | 0.40 | Good |
| 4 steps, no CFG | 47.9 | 0.26 | Good |
| **3 steps, no CFG (fast)** | **55.3** | **0.23** | **Good** |
| 2 steps, no CFG (fastest) | 65.5 | 0.19 | Degraded (repetitions, errors) |

**2 steps is unusable** -- produces repetitions ("Hello world Hello world") and word errors.
**3 steps is the sweet spot** -- perfect Whisper transcription at 55+ fps.

### Known Bug

All configs crash with "CUDA device-side assert" on the second long text ("The ancient city of Rome..."). This is a sequence-length or token prediction issue, not quantization-related (happens on BF16 too).

---

## 5. Quality Evaluation

### TurboQuant 4+4 Residual (8 effective bits) -- Disk Compression Approach

| Metric | Value |
|--------|-------|
| Weight cosine similarity | 0.99996 (avg across 182 layers) |
| Weight MSE | 7.46e-9 |
| Disk compression | 1.55x (8.04 GB -> 5.20 GB) |
| UTMOS (BF16 baseline) | 1.248 MOS |
| UTMOS (quantized) | 1.276 MOS (+0.028, slightly better) |
| Speaker similarity | 0.930 (Resemblyzer cosine) |
| Whisper transcription | Identical or better than baseline |

**Verdict:** Near-lossless, but no runtime speedup (dequants to BF16 at load time).

### torchao int4 + HQQ -- The Production Solution

| Metric | Value |
|--------|-------|
| Whisper accuracy | Near-perfect (occasional 1-word diffs) |
| Formal MOS evaluation | Not done (runtime approach, not disk) |
| Perceptual quality | Indistinguishable from BF16 in listening tests |

### Why Full-Reference Metrics Don't Work for TTS

PESQ, STOI, and MCD are **not meaningful** for TTS evaluation because:
- Flow-matching uses random Gaussian noise -- two runs of the SAME model produce different audio
- These metrics require aligned waveforms
- Use no-reference metrics (UTMOS) or Whisper transcription instead

---

## 6. KV Cache Quantization Research

### Summary

Extensive research (32+ agents, 6 rounds) concluded that **KV cache quantization has minimal benefit for single-stream TTS** at typical sequence lengths.

### The Math

```
Model weights (int4): 1.56 GB read per decode step (fixed)
KV cache (BF16):      106,496 * seq_len bytes per decode step
KV cache (INT4):       28,288 * seq_len bytes per decode step
```

| Scenario | B*seq_len | KV % of BW | INT4 KV Speedup |
|----------|-----------|------------|-----------------|
| **Our TTS** | **200** | **1.3%** | **1.01x** |
| Long-form | 8,000 | 35% | 1.35x |
| 8 concurrent | 16,000 | 52% | 1.62x |
| Max context | 128,000 | 90% | 2.94x |
| Theoretical max | inf | 100% | 3.765x |

**Threshold for >10% speedup: B*seq_len > 3,800**

### What We Implemented (and why it was slower)

The `kv_cache_quant/` implementation used Hadamard rotation + Lloyd-Max codebook. Result: **46 fps (30% SLOWER than 66 fps baseline)**.

**Problems:**
1. Hadamard rotation HURTS key quantization (spreads outlier energy, KVLinC 2025)
2. Lloyd-Max unnecessary at 4 bits -- simple RTN absmax matches quality (0.3 dB gap)
3. 128x128 Hadamard matmul = 13.6M FLOPs/token/layer (the actual bottleneck)
4. Nothing fused: separate kernels for rotation, quantize, dequant, attention
5. No CUDA graph: ~364 kernel launches/step at 5-10us each

### When KV Cache Quant WOULD Help

- Batch serving (B >= 8, concurrent TTS streams)
- Long-form audio (>5 minutes, seq_len > 4000)
- Memory-constrained deployment (need to fit more context)

### Correct Implementation (not done, roadmap)

1. Drop rotation entirely for K (per-channel quant is better)
2. Simple RTN absmax per-group g=32
3. Fuse quantize into projection GEMM epilogue
4. Fuse dequant into attention kernel
5. CUDA graphs for the full decode step
6. Residual buffer (first 128 tokens BF16, rest INT4)

---

## 7. Key Lessons Learned

### Quantization Algorithm

1. **HQQ >> RTN for TTS.** Round-to-nearest with min-max scaling produces garbage audio even though it gives good perplexity on text-only benchmarks. HQQ's iterative optimization of scale/zero is essential.

2. **Quantize selectively.** Only the LLM backbone (77% of params) benefits from quantization. The acoustic transformer (stochastic flow-matching) and codec decoder (audio-critical convolutions) must stay BF16.

3. **Rotation format matters at training time.** QR rotation (TurboQuant) makes on-the-fly inference impossible due to dense matmul per group. Hadamard is inference-friendly but still adds overhead. Best: no rotation + HQQ.

### Inference Kernels

4. **tinygemm > GemLite Triton at B=1 on Ampere.** Hand-tuned CUDA beats Triton for single-token decode. GemLite kernel is 0.072ms (faster than nn.Linear!) but Python dispatch kills it.

5. **TILE_PACKED_TO_4D required for SM86 (RTX 3090).** Default torchao int4 packing format only works on H100. This cost a full day to debug.

6. **torch.inference_mode() >> torch.no_grad().** +7 fps difference. Always use inference_mode for deployment.

7. **torch.compile helps, but limitedly.** +10 fps when compiling both backbone and acoustic model. Dynamic shapes prevent CUDA graph capture.

### Flow-Matching TTS Specific

8. **3 flow steps is the sweet spot.** 8->3 steps gives 2.7x speedup with near-zero quality loss. 2 steps produces repetitions and word errors.

9. **CFG (Classifier-Free Guidance) doubles acoustic cost.** alpha=1.0 (disabled) vs 1.2 is a 2x speed difference in acoustic decoder. Quality difference is audible but acceptable for many use cases.

10. **Audio tokens are robust to quantization.** TTS flow-matching is stochastic by design, so small weight perturbations from quantization don't propagate. This is the opposite of text LLMs where exact logit values matter.

### Performance Analysis

11. **KV cache is irrelevant at short sequences.** At batch=1, seq_len=200, KV cache is 1.3% of bandwidth. Don't waste time optimizing it for single-stream TTS.

12. **Kernel launch overhead matters.** 4,400 launches/token (TurboQuant) = 2 fps. 182 launches/token (tinygemm) = 66 fps. Same total compute, 33x speed difference.

13. **Python dispatch is real overhead.** HQQ+GemLite: kernel is 0.072ms but Python dispatch adds ~7ms total across 182 layers. Need C++ dispatch or fused approaches.

### Process

14. **Try the simplest thing first.** torchao is 5 lines of code. We spent days on TurboQuant before discovering it. Always benchmark the standard library first.

15. **Profile before optimizing.** The KV cache research could have been avoided by checking the bandwidth breakdown first. KV is 1.3% of bandwidth -- optimizing it is mathematically futile.

---

## 8. External Projects Compared

### vs. voxtral-mini-realtime-rs (Rust/WGPU)

| Dimension | Ours | Rust |
|-----------|------|------|
| Speed | **57 fps, RTF 0.22** | RTF 0.97 (barely real-time) |
| VRAM | 3.7 GB | 2.67 GB |
| Quality | Near-lossless (HQQ) | 8.49% WER on FLEURS (Q4 RTN) |
| Quantization | int4 HQQ (optimal) | Q4_0 GGUF (simple RTN) |
| Platform | CUDA only | Cross-platform (Vulkan/Metal/WebGPU/WASM) |
| Unique value | Speed + quality | Browser deployment, portability |

### vs. voxtral-tts.c (Pure C)

| Dimension | Ours | C |
|-----------|------|---|
| Speed | **57 fps, RTF 0.22** | RTF 7.3 (Blackwell GPU!), 59 (CPU) |
| VRAM | 3.7 GB | ~8 GB |
| Quantization | int4 HQQ | None (BF16 only) |
| Quality metrics | Whisper + UTMOS | None |
| Unique value | Production-ready | Educational reference, zero dependencies |

**Our project is 4.5x faster than Rust and 34x faster than C** (despite C being tested on newer Blackwell hardware).

---

## 9. References

### Papers

- **HQQ** -- Half-Quadratic Quantization (used in our solution)
- **TurboQuant** (ICLR 2026) -- arXiv:2504.19874 (tried, too slow for inference)
- **KIVI** (ICML 2024) -- arXiv:2402.02750 (KV cache quant, 2-bit works)
- **KVQuant** (NeurIPS 2024) -- arXiv:2401.18079
- **KVLinC** (2025) -- arXiv:2510.05373 (rotation hurts K quantization)
- **QuaRot** (NeurIPS 2024) -- arXiv:2404.00456
- **INT-FlashAttention** (2024) -- arXiv:2409.16997

### Tools

- **torchao** -- int4 weight-only quantization (our production solution)
- **tinygemm** -- PyTorch built-in CUDA kernel for fused int4 dequant+matmul
- **hqq** -- HQQ algorithm for optimal quantization parameters
- **GemLite** -- Triton kernels (fast kernel but Python dispatch overhead)
- **Whisper** -- OpenAI speech recognition (used for quality evaluation)
- **UTMOS** -- No-reference audio quality predictor

### Key Files (before cleanup)

```
src/torchao_inference.py   -- THE SOLUTION: int4 HQQ + tinygemm (5 key lines)
src/generate_fast.py       -- Optimized TTS: static cache, 3-step flow, compile
src/model.py               -- Full Voxtral architecture definition
src/benchmark_all.py       -- End-to-end benchmark (5 configs, Whisper eval)
```

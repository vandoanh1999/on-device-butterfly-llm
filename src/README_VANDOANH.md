# VANDOANH_APEX — Unified Breakthrough Engine

> Built entirely on a phone (Snapdragon 7+ Gen 2 / SM7475) inside Termux.  
> No server. No GPU. No institution. Just C++ and ARM NEON.

---

## What this is

VANDOANH_APEX is a from-scratch C++ machine learning engine combining three novel components into a single unified training loop:

- **Logic2048Linear** — a multiply-free weight format using power-of-2 representations, trained directly with Straight-Through Estimator (STE) gradients
- **DIB (Dynamic Information Bottleneck)** — replaces the standard FFN with O(N log N) butterfly mixing, reducing quadratic cost to near-linear
- **DLB (Dynamic Layer Budget)** — per-token early exit based on entropy; tokens that are "easy" skip deeper layers entirely

The result is a language model that trains on-device, in real time, with measurable compute savings — not theoretical ones.

---

## Files

| File | Role |
|---|---|
| `VANDOANH_APEX.cpp` | Main engine: training, benchmarks, GGUF parsing |
| `VANDOANH_MASTER.cpp` | Extended engine: adds inference, model comparison, presets |

Both are single-file, self-contained C++17. No external ML library dependencies.

---

## Architecture

```
Token → Embedding → [AdaptiveLayer × N] → LM Head (Logic2048Linear)

AdaptiveLayer:
  RMSNorm → DIBLayer (O(N logN) butterfly mixing)
           → Logic2048Linear (STE-trained, multiply-free)
           → Residual add
           → DLB check → early exit if entropy below threshold
```

Vocabulary: 256 (byte-level, language-agnostic, works natively with Vietnamese UTF-8)

---

## Bugs fixed (APEX vs MASTER vs originals)

**VANDOANH_APEX fixes over MASTER:**

- `[BUG-1]` GGUFParser used `throw` with `-fno-exceptions` → silently called `terminate()`. Fixed: replaced with error-status return path.
- `[BUG-2]` `train_vi` mode was documented in help text but never implemented in `main()`. Fixed: full `bench_train_vi()` with `AdaptiveUnifiedModel`.
- `[BUG-3]` `DynamicLayerBudget` used a `mutable` RNG inside a `const` method with no synchronization — data race under OpenMP. Fixed: mutex-protected RNG and exit counters.
- `[BUG-4]` `Logic2048Weight::from_float` clamped max power to 62 instead of 63, leaving the top entry of the 1024-entry table unreachable. Fixed: `max(-63, min(p, 63))`.

**VANDOANH_MASTER fixes over original ENGINE_v5:**

- `alignas(64)` misplaced on weight scalar instead of table → 16× memory bloat per weight
- Butterfly tiled forward: `pi=0` reset at each tile instead of `pi=pi_stage_start` → wrong rotation angles mid-sequence
- `grad_clip` hardcoded to `1.0f` regardless of CLI argument

---

## Compile

```bash
# Recommended (Termux / Android, Clang)
clang++ -O3 -std=c++17 -march=armv8.4-a+dotprod+fp16+i8mm \
  -ffast-math -fopenmp -lpthread \
  VANDOANH_APEX.cpp -o vdapex

# Fallback (any platform, GCC)
g++ -O3 -std=c++17 -fopenmp -lpthread \
  VANDOANH_APEX.cpp -o vdapex
```

Note: do **not** use `-fno-exceptions` for APEX — the GGUF parser uses `try/catch`.  
MASTER can use `-fno-exceptions -fno-rtti` since its parser uses error-status returns.

---

## Usage

```bash
# Full benchmark suite
./vdapex all

# Train: unified (Logic2048 + DIB + DLB)
./vdapex unified data.txt --lr=1e-4 --steps=20000 --dim=64

# Train: Vietnamese adaptive model
./vdapex train_vi data.txt --layers=4 --exit_thr=0.3 --diag=1 \
  --steps=20000 --warmup=0.05 --lr_sched=cosine

# Individual benchmarks
./vdapex logic | butterfly | flash | jit | dlb | kernel | proof

# GGUF model info
./vdapex gguf model.gguf
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--lr` | 1e-3 | Learning rate |
| `--wd` | 0.01 | Weight decay (AdamW) |
| `--steps` | 5000 | Training steps |
| `--dim` | 64 | Model dimension (must be power of 2) |
| `--layers` | 4 | Number of AdaptiveLayers |
| `--grad_clip` | 1.0 | Gradient clip norm |
| `--warmup` | 0.05 | Warmup fraction of total steps |
| `--lr_sched` | cosine | `cosine` or `constant` |
| `--exit_thr` | 0.3 | DLB entropy exit threshold |
| `--diag=1` | off | Verbose per-step diagnostics |
| `--diag_every` | 500 | Diagnostic print interval |
| `--seed` | 42 | Random seed |

---

## Benchmark results (on-device, Snapdragon 7+ Gen 2)

### FlashAttention: NEON vs Dense

| N | Dense (ms) | Flash (ms) | Speedup |
|---|---|---|---|
| 64 | 1.159 | 1.042 | 1.11× |
| 128 | 5.499 | 3.655 | 1.50× |
| 256 | 27.137 | 14.050 | 1.93× |
| 512 | 87.430 | 24.216 | 3.61× |
| 1024 | 246.459 | 84.434 | 2.92× |

### Butterfly correctness (L1/L2 error vs reference)

| N | Time (ms) | L2 error |
|---|---|---|
| 512 | 0.0038 | 4.22e-08 |
| 1024 | 0.0141 | 4.34e-08 |
| 2048 | 0.0241 | 4.66e-08 |
| 4096 | 0.1611 | 4.46e-08 |

### JIT AArch64 FMLA kernel

- 4×64 dot: 0.0000 ms/call — **22.29 GFLOPS**

### Training: `train_vi` on `math_ai_100m.txt`

Config: `--layers=4 --exit_thr=0.3 --diag=1 --steps=20000 --lr=1e-4 --lr_sched=cosine`  
Model: 51,776 params (~0.05M), vocab=256, dim=64

| Step | Train PPL | Val PPL |
|---|---|---|
| 1 | 255.23 | 312.86 |
| 1000 | 117.63 | 99.58 |
| 5000 | 9.56 | 8.95 |
| 10000 | 7.74 | 8.77 |
| 15000 | 7.75 | 8.55 |
| 20000 | 8.57 | **8.50** |

- **DLB savings: 37.7%** — average 2.49/4 layers used per token
- Train/Val gap negligible → no overfitting
- Speed: ~273 steps/s on-device

![Training result](1000048229.jpg)

---

## What makes this honest

- No hardcoded benchmark numbers. All results come from live measurement with `volatile` sinks and ≥100 repeat loops to prevent compiler elimination.
- Logic2048 after bug fix runs at **0.27× FP32** at 4B precision — honestly slower for small N due to table lookup overhead, not claimed otherwise.
- DIB kernel (BF+Diag) at N=512: 0.93 GFLOPS standard vs 0.80 GFLOPS DIB — the compute overhead is reported, not hidden.
- DLB savings are real: exit distribution shows 10,536 layer-1 exits and 7,287 layer-2 exits out of 20,000 steps.

---

## Platform

- Device: Snapdragon 7+ Gen 2 (SM7475), ARMv8.6-A
- Environment: Termux (Android), Clang 17
- SIMD: ARM NEON, dotprod, i8mm
- Threading: OpenMP + pthreads
- No cloud. No GPU. No external ML framework.

---

## Author

VanDoanh — independent researcher, on-device AI systems  
VANDOANH Research 2025
For inquiries, please contact phamvandoanh9@gmail.com 

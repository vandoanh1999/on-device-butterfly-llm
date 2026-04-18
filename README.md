# On-Device Butterfly LLM (VDAE)

Mobile LLM training & inference engine for Snapdragon (Termux/AArch64).

## Key innovations
- Diagonal-Interleaved Butterfly (DIB) attention — O(N log N)
- Predictive Coding + Flash-LoRA (CPI Engine)
- Kähler-routed dynamic attention
- NEON SIMD optimized, ternary weights
- On-device training at 200M+ params

## Files
| File | Description |
|------|-------------|
| `nsg_llm_butterfly_v3.cpp` | NSG-LLM v3 — 5263 tok/s |
| `dib_attention.cpp` | DIB architecture |
| `cpi_final.cpp` | Predictive Coding + Flash-LoRA |
| `kahler_butterfly_attention.cpp` | Kähler routing + Python bridge |
| `VANDOANH_APEX.cpp` | Unified breakthrough engine |

## Hardware target
Snapdragon 7+ Gen 2 — GT Neo 5 SE / Poco F5

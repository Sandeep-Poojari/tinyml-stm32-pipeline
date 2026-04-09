# STM32 TinyML Benchmark

## Platform
- MCU: STM32L4 (Cortex-M4 @ 120 MHz)
- Toolchain: STM32CubeMX + X-CUBE-AI

## Model
- Dataset: MNIST
- Architecture: Small CNN (Conv → Pool → Conv → Pool → Dense)

---

## FLOAT32 Model

- Latency: 26.7 ms per inference
- Weights: ~31 KB
- RAM (activations): ~8.3 KB
- CPU cycles: ~3.2M
- Cycles/MACC: ~15

---

## INT8 Quantized Model

- Latency: 8.6 ms per inference
- Weights: 7.8 KB
- RAM (activations): 4.6 KB
- CPU cycles: ~1.03M
- Cycles/MACC: ~5.1

---

## Improvements

- 🚀 ~3.1x faster inference
- 💾 ~75% smaller model size
- 📉 ~45% reduction in RAM usage
- ⚡ ~3x better compute efficiency

---

## Notes

- Quantization: Full integer (INT8)
- Input/Output: INT8 with scaling
- Validation performed on target hardware using X-CUBE-AI
- Accuracy validated via cross-check (no degradation observed in test run)

---

## Key Insight

Convolution layers dominate runtime (>95%), making them the primary optimization target.
INT8 quantization significantly improves performance due to efficient integer MAC operations.
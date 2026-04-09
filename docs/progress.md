

# Project Progress

## Goal

Build an end-to-end TinyML pipeline:
Keras → TFLite → STM32 (X-CUBE-AI) → On-device validation

---

## Completed Steps

### 1. Model Training
- Dataset: MNIST
- Framework: TensorFlow / Keras
- Achieved ~98% test accuracy

### 2. Model Conversion
- Converted Keras model → TFLite
- Implemented post-training quantization (INT8)

### 3. Quantization
- Full integer quantization (weights + activations)
- Reduced model size significantly (~75%)

### 4. STM32 Integration
- Imported model into STM32CubeMX
- Generated code using X-CUBE-AI
- Built and flashed on STM32L4 board

### 5. On-Device Validation
- Ran inference on target hardware
- Verified correctness via cross-validation (100% match between reference and embedded model)

### 6. Performance Profiling
- Measured latency, memory, and CPU cycles
- Compared FLOAT vs INT8 models

---

## Current Status

- End-to-end pipeline is fully functional
- INT8 model successfully deployed and validated on hardware
- Benchmarking completed (see benchmark.md)

---

## Next Steps

- Validate with real dataset (not random input)
- Optimize model architecture (reduce latency further)
- Measure power consumption on target
- Integrate real sensor or image input pipeline

---

## Key Learnings

- Quantization provides major performance gains on embedded targets
- Convolution layers dominate runtime and are key optimization targets
- Memory footprint is significantly reduced with INT8 models
- Toolchain integration (CubeMX + X-CUBE-AI) requires careful setup but is powerful once working
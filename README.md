# TinyML STM32 Pipeline 🚀

> End-to-end pipeline to train, quantize, and deploy ML models on STM32 microcontrollers using TensorFlow Lite and X-CUBE-AI.

---

## ✨ Why this project matters

Running ML on microcontrollers requires tight control over **latency, memory, and power**.  
This project demonstrates a **production-style workflow** to:

- Shrink models using **INT8 quantization**
- Deploy on **resource-constrained hardware (STM32 Cortex-M4)**
- Validate and benchmark **on real hardware**

---

## 🎯 Goal

Build a reproducible workflow to:
- Train a lightweight neural network
- Convert to TensorFlow Lite (TFLite)
- Apply **full integer (INT8) quantization**
- Deploy on **STM32 (X-CUBE-AI)**
- Benchmark **latency, RAM, Flash, and accuracy**

---

## 🧠 Architecture Overview

```
          +-------------------+
          |   Dataset (MNIST) |
          +---------+---------+
                    |
                    v
          +-------------------+
          |  Train (Keras)    |
          |  -> .keras model  |
          +---------+---------+
                    |
                    v
          +-------------------+
          | TFLite Convert    |
          | + INT8 Quantize   |
          | -> .tflite        |
          +---------+---------+
                    |
                    v
          +-------------------+
          | STM32CubeMX       |
          | + X-CUBE-AI       |
          | Code Generation   |
          +---------+---------+
                    |
                    v
          +-------------------+
          |  STM32 Hardware   |
          |  Inference + Prof |
          +-------------------+
```

---

## 🧩 Tech Stack

- **Python**: TensorFlow, NumPy, Matplotlib
- **TFLite**: Model conversion & quantization
- **STM32**: STM32CubeMX + X-CUBE-AI
- **Embedded C**: Deployment & inference

---

## 📁 Project Structure

```
tinyml-stm32-pipeline/
├── src/           # training, conversion, evaluation scripts
├── models/        # trained (.keras) and quantized (.tflite)
├── stm32/         # STM32CubeMX project + generated code
├── docs/          # benchmark, progress, architecture notes
├── notebooks/     # experiments and exploration
```

---

## 🚀 Getting Started

### 1. Create environment
```bash
conda create -n tinyml python=3.10
conda activate tinyml
pip install -r requirements.txt
```

> **Apple Silicon (M-series) users**:
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## ▶️ Run

### Train model
```bash
python src/train/train_mnist.py
```

### Convert to TFLite (INT8)
```bash
python src/convert/convert_to_tflite.py
```

---

## 🧪 Deployment (STM32)

1. Open **STM32CubeMX**
2. Enable **X-CUBE-AI**
3. Import `.tflite` model
4. Generate project
5. Build & flash (STM32CubeIDE or VS Code extension)
6. Run validation on target (UART / ST-LINK)

**Tested on:**
- STM32L4 (Cortex-M4 @ 120 MHz)

---

## 📊 Benchmark Results

See detailed results:
- 📄 [Benchmark](docs/benchmark.md)
- 📄 [Progress](docs/progress.md)

### Summary

| Metric        | FP32 Model | INT8 Model |
|---------------|-----------|------------|
| Latency       | ~26.7 ms  | ~8.6 ms    |
| Model Size    | ~31 KB    | ~7.8 KB    |
| RAM Usage     | ~8.3 KB   | ~4.6 KB    |
| Efficiency    | ~15 cyc/MACC | ~5 cyc/MACC |

---

## 📌 Status

- [x] Train baseline model (MNIST CNN)
- [x] Convert to TFLite
- [x] Apply INT8 quantization
- [x] Deploy on STM32 (STM32L4)
- [x] On-device validation & profiling
- [ ] Real dataset validation on-device
- [ ] Power profiling

---

## 💡 Key Insights

- INT8 quantization provides ~3x speedup on Cortex-M4
- Model size reduced by ~75% with no observed accuracy drop (test setup)
- Convolution layers dominate runtime (>95%)
- Integer MAC operations significantly improve efficiency
- Hardware-aware optimization is critical for TinyML

---

## 🔮 Future Work

- Quantization-aware training (QAT)
- CMSIS-NN optimization
- Sensor-based models (IMU/audio)
- Real-time inference pipelines
- Power/energy benchmarking

---

## 🤝 Contributing

Contributions are welcome. Feel free to:
- Improve models or pipeline
- Add new datasets
- Optimize embedded performance

---

## 📜 License

- MIT License (for this project)
- STM32 generated code is subject to STMicroelectronics license terms
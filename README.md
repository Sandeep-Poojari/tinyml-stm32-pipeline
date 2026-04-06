# TinyML STM32 Pipeline 🚀

> End-to-end pipeline to train, quantize, and deploy ML models on STM32 microcontrollers using TensorFlow Lite.

---

## 🎯 Goal
Build a reproducible workflow to:
- Train a lightweight neural network
- Convert to TensorFlow Lite (TFLite)
- Apply **full integer (INT8) quantization**
- Deploy on **STM32 (X-CUBE-AI)**
- Benchmark **latency, RAM, Flash, and accuracy**

---

## 🧠 Pipeline Overview

```
Dataset → Training → Model (.h5/.keras)
        → TFLite Conversion → Quantized Model (.tflite)
        → STM32 Deployment → Inference → Benchmarking
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
├── models/        # trained (.h5/.keras) and quantized (.tflite)
├── stm32/         # STM32CubeMX project + generated code
├── docs/          # architecture and design docs
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

## 📊 Results (example)

| Metric        | FP32 Model | INT8 Model |
|---------------|-----------|------------|
| Model Size    | ~80 KB    | ~20 KB     |
| RAM Usage     | High      | Low        |
| Latency       | Higher    | Lower      |
| Accuracy      | ~99%      | ~97–99%    |

> Results will vary depending on model and STM32 target.

---

## 🧪 Deployment (STM32)

1. Open **STM32CubeMX**
2. Enable **X-CUBE-AI**
3. Import `.tflite` model
4. Generate project
5. Flash to board
6. Measure performance (latency, RAM, Flash)

---

## 📌 Status

- [x] Project structure created
- [ ] Train baseline model
- [ ] Convert to TFLite
- [ ] Apply INT8 quantization
- [ ] Deploy on STM32
- [ ] Benchmark performance

---

## 🔮 Future Work

- Quantization-aware training (QAT)
- CMSIS-NN optimization
- Sensor-based models (IMU/audio)
- Real-time inference pipelines

---

## 📜 License

MIT
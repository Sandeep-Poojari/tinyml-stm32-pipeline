
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------------------
# Config
# -------------------------------

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models/trained")
TFLITE_DIR = os.path.join(PROJECT_ROOT, "models/tflite")

KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.keras")
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, "mnist_model_int8.tflite")

# -------------------------------
# Load dataset (for calibration)
# -------------------------------
(x_train, _), _ = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, -1)

# -------------------------------
# Representative dataset
# -------------------------------
def representative_data_gen():
    for i in range(100):
        yield [x_train[i:i+1]]

# -------------------------------
# Load model
# -------------------------------
model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

 # -------------------------------
# Convert to TFLite (INT8)
# -------------------------------
 # Save as SavedModel (more stable for conversion)
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/saved_model")
model.export(SAVED_MODEL_DIR)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Allow fallback to builtins for stability
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# -------------------------------
# Convert
# -------------------------------
tflite_model = converter.convert()

# -------------------------------
# Save model
# -------------------------------
os.makedirs(TFLITE_DIR, exist_ok=True)

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("\n✅ TFLite model generated successfully!")
print(f"Saved to: {TFLITE_MODEL_PATH}")

# -------------------------------
# Check model size
# -------------------------------
keras_size = os.path.getsize(KERAS_MODEL_PATH) / 1024
int8_size = os.path.getsize(TFLITE_MODEL_PATH) / 1024

print(f"\n📦 Model Size Comparison:")
print(f"Keras model: {keras_size:.2f} KB")
print(f"INT8 TFLite model: {int8_size:.2f} KB")


import tensorflow as tf
import numpy as np
import os

# -------------------------------
# Config
# -------------------------------

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models/trained")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.keras")

EPOCHS = 5
BATCH_SIZE = 32

# -------------------------------
# Load dataset
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Add channel dimension (28x28 → 28x28x1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# -------------------------------
# Model (Tiny + Quantization-friendly)
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# -------------------------------
# Compile
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train
# -------------------------------
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# -------------------------------
# Evaluate
# -------------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# -------------------------------
# Save model
# -------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
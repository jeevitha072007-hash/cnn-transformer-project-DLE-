import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10

print("TensorFlow version:", tf.__version__)

# ============================================================
# 1. Load CIFAR-10 Dataset
# ============================================================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)


# ============================================================
# 2. CNN Feature Extraction
# ============================================================

inputs = layers.Input(shape=(32, 32, 3))

x = layers.Conv2D(
    32, (3, 3),
    activation="relu",
    padding="same"
)(inputs)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(
    64, (3, 3),
    activation="relu",
    padding="same"
)(x)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(
    128, (3, 3),
    activation="relu",
    padding="same"
)(x)

print("CNN feature map:", x.shape)


# ============================================================
# 3. Convert CNN Feature Map into Tokens
# ============================================================

# 8 × 8 feature map → 64 tokens
x = layers.Reshape((64, 128))(x)

print("Token representation:", x.shape)


# ============================================================
# 4. Transformer Encoder
# ============================================================

attention_output = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(x, x)

# Residual connection
x = layers.Add()([x, attention_output])

# Layer normalization
x = layers.LayerNormalization()(x)


# Feed Forward Network
ffn = layers.Dense(256, activation="relu")(x)
ffn = layers.Dense(128)(ffn)

# Residual connection
x = layers.Add()([x, ffn])

# Layer normalization
x = layers.LayerNormalization()(x)

print("Transformer output:", x.shape)


# ============================================================
# 5. Classification Head
# ============================================================

x = layers.GlobalAveragePooling1D()(x)

print("Global feature:", x.shape)

outputs = layers.Dense(
    10,
    activation="softmax"
)(x)


# ============================================================
# 6. Create Model
# ============================================================

model = Model(inputs, outputs)

model.summary()


# ============================================================
# 7. Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. Train Model
# ============================================================

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64
)


# ============================================================
# 9. Evaluate Model
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n====================================")
print("FINAL TEST ACCURACY:", test_accuracy)
print("FINAL TEST LOSS:", test_loss)
print("====================================")
import matplotlib.pyplot as plt

# Training Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Training Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
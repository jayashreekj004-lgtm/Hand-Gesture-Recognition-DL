import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# -----------------------------
# 1. Dataset path
# -----------------------------
dataset_path = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

# -----------------------------
# 2. Load training dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# 3. Load validation dataset
# -----------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nClasses:")
print(train_ds.class_names)

# -----------------------------
# 4. Improve performance
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 5. Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# -----------------------------
# 6. MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# 7. Build model
# -----------------------------
model = models.Sequential([
    data_augmentation,

    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        5,
        activation="softmax"
    )
])

# -----------------------------
# 8. Compile model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 9. Model summary
# -----------------------------
model.summary()

# -----------------------------
# 10. Train model
# -----------------------------
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# 11. Save model
# -----------------------------
import os

os.makedirs("models", exist_ok=True)

model.save(
    "models/hand_gesture_model.keras"
)

print("\nModel training completed!")
print("Model saved successfully!")
print("Location: models/hand_gesture_model.keras")
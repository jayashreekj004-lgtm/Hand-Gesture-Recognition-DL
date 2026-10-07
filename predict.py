import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "models/hand_gesture_model.keras"
)

# Class names
class_names = [
    "01_palm",
    "03_fist",
    "05_thumb",
    "06_index",
    "07_ok"
]

# Image path
image_path = "test_images/fist.png"

# Load image
img = Image.open(image_path).convert("RGB")

# Resize image
img = img.resize((224, 224))

# Convert image to array
img_array = np.array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

# Get predicted class
predicted_index = np.argmax(prediction[0])

# Get confidence
confidence = prediction[0][predicted_index] * 100

print("\nPredicted Gesture:", class_names[predicted_index])
print("Confidence:", round(confidence, 2), "%")
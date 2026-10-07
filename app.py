import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "models/hand_gesture_model.keras"
)

# Class names — same order as training
class_names = [
    "Palm ✋",
    "Fist ✊",
    "Thumbs Up 👍",
    "Index Finger ☝️",
    "OK 👌"
]


def predict_gesture(image):

    # Same preprocessing as predict.py
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)

    # Same as predict.py
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Find highest probability
    predicted_index = np.argmax(prediction[0])

    # Confidence
    confidence = prediction[0][predicted_index] * 100

    result = (
        f"Gesture: {class_names[predicted_index]}\n"
        f"Confidence: {confidence:.2f}%"
    )

    return result


# Gradio interface
interface = gr.Interface(
    fn=predict_gesture,
    inputs=gr.Image(
        type="pil",
        label="Upload Hand Gesture Image"
    ),
    outputs=gr.Textbox(
        label="Prediction"
    ),
    title="Hand Gesture Recognition System",
    description="Upload a hand gesture image to identify the gesture."
)

# Start application
interface.launch()
# Hand Gesture Recognition System Using Deep Learning and Transfer Learning

## Project Overview

This project is a deep learning-based Hand Gesture Recognition System that classifies hand images into predefined gesture categories.

The system uses Transfer Learning with MobileNetV2 to recognize different hand gestures and provides the prediction with confidence through a Gradio web interface.

## Gestures Recognized

The model recognizes 5 hand gestures:

- Palm ✋
- Fist ✊
- Thumbs Up 👍
- Index Finger ☝️
- OK 👌

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Deep Learning
- Transfer Learning
- NumPy
- Pillow
- Gradio

## Dataset

The project uses the Hand Gesture Recognition Database dataset.

Dataset source:

Kaggle - Hand Gesture Recognition Database

For this project, 5 gesture classes were selected:

```text
01_palm
03_fist
05_thumb
06_index
07_ok
A total of 5,000 images were used:

1,000 Palm images
1,000 Fist images
1,000 Thumbs Up images
1,000 Index Finger images
1,000 OK images

Project Workflow

Hand Gesture Image
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
MobileNetV2
        ↓
Feature Extraction
        ↓
Gesture Classifier
        ↓
Gesture Prediction
        ↓
Gradio Interface

Model

MobileNetV2 is used as the base model for Transfer Learning.

The pretrained ImageNet weights are used for feature extraction.

The classification layers are added on top of MobileNetV2 to classify the 5 selected gestures.

Image Preprocessing

Images are resized to:

224 × 224

MobileNetV2 preprocessing is applied to convert pixel values into the required range.

Model Training

The dataset was divided into:

80% Training data
20% Validation data

The model was trained for 10 epochs.

The trained model is saved as:
models/hand_gesture_model.keras
Gradio Interface

A Gradio interface is created for easy testing.

Users can upload a hand gesture image and the system displays:
Gesture
Confidence

Example:

Gesture: Fist ✊
Confidence: 86.19%
Hand-Gesture-Recognition-DL/
│
├── dataset/
│   ├── 01_palm/
│   ├── 03_fist/
│   ├── 05_thumb/
│   ├── 06_index/
│   └── 07_ok/
│
├── models/
│   └── hand_gesture_model.keras
│
├── test_images/
│
├── extract_dataset.py
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
How to Run
1. Create Virtual Environment

python -m venv venv

2. Activate Virtual Environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1

3. Install Requirements

pip install -r requirements.txt


4. Train the Model

python train_model.py

5. Test Prediction
python predict.py

6. Run Gradio Application
python app.py
Then open:

http://127.0.0.1:7860


Sample Results

The trained model was manually tested with the five gesture classes.

Gesture	Prediction	Confidence
Palm ✋	Palm ✋	85.11%
Fist ✊	Fist ✊	86.19%
Thumbs Up 👍	Thumbs Up 👍	99.99%
Index Finger ☝️	Index Finger ☝️	95.28%
OK 👌	OK 👌	100.00%

These values are from a small manual test and should not be interpreted as the model's overall test-set accuracy.

Applications

This system can be used as a basic foundation for:

Touchless computer interaction
Gesture-based gaming
Human-computer interaction
Smart home control
Assistive technology
Gesture-controlled applications
Future Improvements
Add more gesture classes
Use a larger and more diverse dataset
Add real-time webcam gesture detection
Improve performance on real-world images
Add gesture-based computer controls
Deploy the application online
Conclusion

This project demonstrates how Deep Learning and Transfer Learning can be used to build a Hand Gesture Recognition System.

MobileNetV2 is used to extract image features and classify hand gestures into predefined categories. A Gradio interface provides a simple way for users to upload images and view the predicted gesture and confidence score.

## Project Screenshot
https://github.com/jayashreekj004-lgtm/Hand-Gesture-Recognition-DL/blob/main/gesture_prediction.png

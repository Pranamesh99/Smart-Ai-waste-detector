import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "waste_classifier.h5"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Train the model first.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (ensure these match the order during training)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Streamlit UI
st.title("♻️ Intelligent Waste Sorting System")
st.write("Upload an image of waste, and the AI will classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((150, 150))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make a prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display result
    st.success(f"Predicted Category: **{predicted_class.capitalize()}**")
    st.info(f"Confidence: {confidence:.2f}%")

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Use a raw string for the file path so that backslashes are handled correctly.
MODEL_PATH = r"C:\Users\karthik\OneDrive\Desktop\Waste\waste_classifier.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define your class labels in the correct order (must match training)
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.route('/')
def index():
    return "Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check that a file is in the POST request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Open the image file, convert it to RGB, and resize it
        img = Image.open(file).convert("RGB")
        img = img.resize((150, 150))
        # Convert the image to a NumPy array and add a batch dimension
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize the image if your model expects values between 0 and 1
        img_array = img_array / 255.0

        # Make a prediction with your model
        prediction = model.predict(img_array)
        # Get the index of the highest prediction probability
        pred_class = int(np.argmax(prediction, axis=1)[0])
        predicted_label = CLASS_LABELS[pred_class]

        # Return the predicted label as "category" in the JSON response
        return jsonify({"category": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

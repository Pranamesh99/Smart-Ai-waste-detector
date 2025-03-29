from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Waste Sorting API"}
# 🔹 Load your trained model
MODEL_PATH = "waste_classifier.h5"  # Change to your actual model path
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Load and preprocess image
    image = Image.open(file_path).convert("RGB")
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Debug log
    print("Image shape:", image_array.shape)
    
    predictions = model.predict(image_array)
    print("Raw predictions:", predictions)
    
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = np.max(predictions) * 100

    print("Predicted class:", predicted_class, "with confidence:", confidence)
    
    return jsonify({"category": predicted_class})


    return {"prediction": predicted_class}

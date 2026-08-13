from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid

# -----------------------------
# Initialize Flask
# -----------------------------

app = Flask(__name__)
CORS(app)

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "pneumonia_model.keras"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# Load CNN Model
# -----------------------------

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ CNN model loaded successfully.")
except Exception as e:
    model = None
    print("❌ Error loading model:", e)

# -----------------------------
# Home Route
# -----------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Pneumonia Detection API is running successfully!"
    })

# -----------------------------
# Prediction Route
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check uploaded file
    if "image" not in request.files:
        return jsonify({
            "error": "No image file uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    # Check model
    if model is None:
        return jsonify({
            "error": "Model could not be loaded."
        }), 500

    try:

        # -----------------------------
        # Save Uploaded Image
        # -----------------------------

        extension = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + extension

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        # -----------------------------
        # Image Preprocessing
        # -----------------------------

        image = Image.open(filepath).convert("RGB")

        # Model input size
        image = image.resize((224, 224))

        image_array = np.array(image)
        image_array = image_array.astype("float32") / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(image_array)

        probability = float(prediction[0][0])

        if probability >= 0.5:
            result = "Pneumonia"
            confidence = probability * 100
        else:
            result = "Normal"
            confidence = (1 - probability) * 100

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
import os

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "car_bike_model.keras")

IMG_SIZE = (224, 224)

# ==============================
# LOAD MODEL
# ==============================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ==============================
# FLASK APP
# ==============================

app = Flask(__name__)

# ==============================
# HOME ROUTE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# PREDICTION ROUTE
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image
    img_path = os.path.join(BASE_DIR, "temp.jpg")
    file.save(img_path)

    # Load & preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = float(model.predict(img, verbose=0)[0][0])

    if prediction > 0.5:
        label = "CAR"
        confidence = prediction
    else:
        label = "BIKE"
        confidence = 1 - prediction

    # Cleanup
    os.remove(img_path)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })

# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

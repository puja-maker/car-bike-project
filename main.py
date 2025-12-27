import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template

# -------------------------------
# Paths & Config
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "car_bike_model.keras")
IMG_SIZE = (224, 224)

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Model (with logging)
# -------------------------------
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    raise e

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    # index.html MUST be inside templates/
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Azure-safe temp path
    img_path = "/tmp/temp.jpg"
    file.save(img_path)

    try:
        # Preprocess image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img = image.img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = float(model.predict(img, verbose=0)[0][0])

        label = "CAR" if prediction > 0.5 else "BIKE"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


# -------------------------------
# Local Run Only (Gunicorn ignores this)
# -------------------------------
if __name__ == "__main__":
    app.run()

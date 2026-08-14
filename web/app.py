from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import requests
import os

app = Flask(__name__)

model_url = "https://huggingface.co/bugbug05/CNN_plant_disease_model/resolve/main/plant_disease_model_v2.keras"
if not os.path.exists("model.keras"):
    r = requests.get(model_url)
    with open("model.keras", "wb") as f:
        f.write(r.content)

model = load_model("model.keras")

# Danh sách class
classes = [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_apple_rust",
    "Apple_healthy",
    "Blueberry_healthy",
    "Cherry_powdery_mildew",
    "Cherry_healthy",
    "Corn_leaf_spot",
    "Corn_rust",
    "Corn_leaf_blight",
    "Corn_healthy",
    "Grape_black_rot",
    "Grape_black_measles",
    "Grape_leaf_blight",
    "Grape_healthy",
    "Orange_citrus_greening",
    "peach_bacterial_spot",
    "peach_healthy",
    "pepper_bacterial_spot",
    "pepper_healthy",
    "potato_early_blight",
    "potato_late_blight",
    "potato_healthy",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_powdery_mildew",
    "Strawberry_leaf_scorch",
    "Strawberry_healthy",
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites",
    "Tomato_target_spot",
    "Tomato_yellow_leaf_curl_virus",
    "Tomato_moisaic_virus",
    "Tomato_healthy"
]

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            img = keras_image.load_img(filepath, target_size=(150, 150))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            pred = model.predict(x)

            predicted_index = np.argmax(pred)

            prediction = classes[predicted_index]

            confidence = float(np.max(pred) * 100)

            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True, port = 8080)

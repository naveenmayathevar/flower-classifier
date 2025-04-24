
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and class names
model = load_model('mobilenetv2_best.h5')
with open("class_names.json", "r") as f:
    class_names = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            filename = img_file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(path)

            # Preprocess image
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            class_idx = np.argmax(preds)
            confidence = round(100 * np.max(preds), 2)
            prediction = f"{class_names[class_idx]} ({confidence}%)"

    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)

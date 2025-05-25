from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import gdown

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# File download setup
MODEL_URL_ID = "1pBhNYNUDbssgD4wTi32c-pL2VkN9ZtiQ"
CLASS_JSON_ID = "1IpwcBjs-NXLMmqPaH21c0TGjOCF-kXhA"

# Ensure model file is available
if not os.path.exists("mobilenetv2_best.h5"):
    gdown.download(id=MODEL_URL_ID, output="mobilenetv2_best.h5", quiet=False)

# Ensure class names file is available
if not os.path.exists("class_names.json"):
    gdown.download(id=CLASS_JSON_ID, output="class_names.json", quiet=False)

# Load model and class labels
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
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

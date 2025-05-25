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

# Download model and class_names.json
model_path = "mobilenetv2_best.h5"
class_path = "class_names.json"

if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=1pBhNYNUDbssgD4wTi32c-pL2VkN9ZtiQ", model_path, quiet=False)

if not os.path.exists(class_path):
    gdown.download("https://drive.google.com/uc?id=1IpwcBjs-NXLMmqPaH21c0TGjOCF-kXhA", class_path, quiet=False)

model = load_model(model_path)

with open(class_path, "r") as f:
    class_names = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        try:
            img_file = request.files["image"]
            if img_file:
                filename = img_file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img_file.save(path)

                img = image.load_img(path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)
                class_idx = np.argmax(preds)
                confidence = round(100 * np.max(preds), 2)
                prediction = f"{class_names[class_idx]} ({confidence}%)"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

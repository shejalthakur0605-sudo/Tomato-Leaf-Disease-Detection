from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model("tomato_disease_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    solution = ""

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        if prediction[0][0] <= 0.5:
            result = "Early Blight Detected"
            solution = (
                "Recommended Actions:\n"
                "- Remove infected leaves\n"
                "- Use Mancozeb or copper fungicide\n"
                "- Avoid overhead irrigation\n"
                "- Improve air circulation"
            )
        else:
            result = "Healthy Leaf"
            solution = "No disease detected. Maintain good farming practices."

    return render_template("index.html", result=result, solution=solution)

if __name__ == "__main__":
    app.run(debug=True)
import io

import flask
import numpy as np
from flask import Flask, jsonify
from PIL import Image

from classifier import Classifier

app = Flask(__name__)

clf = Classifier("SimpleNet-29-0.41.hdf5", "cifar10.json")


@app.route("/ready", methods=["GET"])
def check_connection():
    """End point to check connection
    """
    return jsonify({"status": "ready"})


@app.route("/predict", methods=["POST"])
def predict():
    """End point for image to be posted
    """
    ret = {"success": False, "prediction": None}

    if flask.request.method == "POST":
        # try:
        image_bytes = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.asarray(image, dtype="int32")

        pred = clf.predict(image_np)
        ret["prediction"] = pred
        ret["success"] = True
        # except Exception as e:
        #     print("Error: ", e)
        #     ret["error"] = e

    else:
        print("Route only accepts POST")

    return jsonify(ret)


if __name__ == "__main__":
    app.run(host="0.0.0.0")

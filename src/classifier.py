import json

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model


class Classifier(object):
    def __init__(self, weight_path, classes_path):
        self.weight_path = weight_path
        self.classes_path = classes_path
        self.clf = load_model(
            self.weight_path,
            custom_objects={
                "GlorotUniform": tf.keras.initializers.glorot_uniform
            },
        )
        with open(self.classes_path) as f:
            self.class_dict = json.load(f)
        self.num_class = len(self.class_dict)
        self.input_size = self.clf.input_shape[1:-1]

    def predict(self, image):
        """Returns the classification result for a given image
        Input: np.array RGB
        """

        processed_image = np.expand_dims(
            np.array(
                Image.fromarray(image.astype(np.uint8)).resize(
                    self.input_size
                ),
                dtype=np.float16,
            ),
            axis=0,
        )/255

        preds = self.clf.predict(processed_image)
        pred_idx = np.argmax(preds[0], axis=-1)

        return self.class_dict[str(pred_idx)]


if __name__ == "__main__":
    test = Classifier("SimpleNet-29-0.41.hdf5", "cifar10.json")

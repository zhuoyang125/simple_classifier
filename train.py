from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from tensorflow import keras
from utils import get_data
from model import get_CNN
from evaluation import evaluate_model
import os
import argparse

X_train,Y_train,X_test,Y_test = get_data()
model = get_CNN

history = model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=20,
                    validation_split=0.1)


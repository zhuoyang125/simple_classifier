# -*- coding: utf-8 -*-
"""CIFAR-10 SimpleNet Data Aug Parameter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YGklQ971c5nNad5pZwdYVu2JvDSrxqiW
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data ():
    """
    Data returning function 
    returns:
    cifar-10 dataset (X_train, Y_train, X_test, Y_test)
    """
    (X_train_orig,Y_train_orig),(X_test_orig,Y_test_orig) = cifar10.load_data()
    #normalize
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
  
  
    #one-hot encoding
    Y_train = to_categorical(Y_train_orig, 10)
    Y_test = to_categorical(Y_test_orig, 10)
    return X_train, Y_train, X_test, Y_test

inputs = keras.Input(shape=(32,32,3))
#1st layer
x = layers.Conv2D(64,3,padding='same',kernel_initializer='he_uniform')(inputs)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

#2nd, 3rd, 4th layer w non-overlapping maxpooling
x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.095)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)

x = layers.MaxPooling2D(pool_size=2,strides=2)(x)

#5th, 6th layer
x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

#7th layer w non-overlapping maxpooling
x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.MaxPooling2D(pool_size=2,strides=2)(x)

#8th, 9th layer w non-overlapping maxpooling
x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.MaxPooling2D(pool_size=2,strides=2)(x)

#10th, 11th, 12th layer w non-overlapping maxpooling
x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128,1,kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128,1,kernel_initializer='he_uniform')(x)
x = layers.BatchNormalization(axis=-1, momentum=0.95)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.MaxPooling2D(pool_size=2,strides=2)(x)

#13th layer
x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)

#flatten
x = layers.Flatten()(x)

outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs,outputs=outputs,name='simplenet')
model.summary()

X_train,Y_train,X_test,Y_test=data()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=20,
                    validation_split=0.1)
test_scores = model.evaluate(X_test, Y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

datagen = ImageDataGenerator(width_shift_range=0.2,
                             horizontal_flip=True,
                             height_shift_range=0.2,
                             rotation_range=20)
datagen.fit(X_train)

#returns augmented images in batches

model.fit_generator(datagen.flow(X_train,Y_train,batch_size=64), 
                    steps_per_epoch=X_train.shape[0]//64,
                    epochs=20,
                    validation_data=(X_test,Y_test)
                    )

test_scores = model.evaluate(X_test, Y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

pred=model.predict(X_test)

import numpy as np
pred = np.argmax(pred, axis=1)

label = np.argmax(Y_test,axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label, pred)
cm


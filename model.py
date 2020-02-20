# -*- coding: utf-8 -*-
"""CIFAR-10 SimpleNet Data Aug Parameter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YGklQ971c5nNad5pZwdYVu2JvDSrxqiW

    @article{hasanpour2016lets,
  title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
  author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:1608.06037},
  year={2016}
}
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow import keras




def get_SimpleNet(config):
    
    
    name = config['name']
    input_shape = config['input_shape']
    momentum = config['momentum']
    dropout_rate = config['dropout_rate']
    filter_dims = config['filter_dims']
    num_class = config['num_class']
    
    inputs = keras.Input(shape=(input_shape))
    
    for i, dims in enumerate(filter_dims):
        if i == 0:
            x = layers.Conv2D(dims,3,padding='same',kernel_initializer='he_uniform')(inputs)
            x = layers.BatchNormalization(axis=-1, momentum=momentum)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
        else:
            x = layers.Conv2D(dims,3,padding='same',kernel_initializer='he_uniform')(x)
            x = layers.BatchNormalization(axis=-1, momentum=momentum)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)

            x = layers.Conv2D(dims,3,padding='same',kernel_initializer='he_uniform')(x)
            x = layers.BatchNormalization(axis=-1, momentum=momentum)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)

            x = layers.Conv2D(dims,3,padding='same',kernel_initializer='he_uniform')(x)
            x = layers.BatchNormalization(axis=-1, momentum=momentum)(x)
            x = layers.Activation('relu')(x)

            x = layers.MaxPooling2D(pool_size=2,strides=2)(x)
    
    
    #13th layer
    x = layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform')(x)

    #flatten
    x = layers.Flatten()(x)

    outputs = layers.Dense(num_class, activation='softmax')(x)

    model = keras.Model(inputs=inputs,outputs=outputs,name=name)

    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

    
   
    


    

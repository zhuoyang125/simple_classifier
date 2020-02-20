import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import json



def get_cifar10(config):
    """
    Data returning function 
    returns:
    cifar-10 dataset (X_train, Y_train, X_test, Y_test)
    """
    num_class = config['num_class']
    (X_train_orig,Y_train_orig),(X_test_orig,Y_test_orig) = cifar10.load_data()
    #normalize
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
  
  
    #one-hot encoding
    Y_train = to_categorical(Y_train_orig, num_class)
    Y_test = to_categorical(Y_test_orig, num_class)
    return X_train, Y_train, X_test, Y_test

def get_mnist(config):
    """
    Data returning function 
    returns:
    MNIST dataset (X_train, Y_train, X_test, Y_test)
    """
    num_class = config['num_class']
    (X_train_orig,Y_train_orig),(X_test_orig,Y_test_orig) = mnist.load_data()
    #normalize
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    #add color channel
    X_train = (tf.expand_dims(X_train, 3))
    X_test = (tf.expand_dims(X_test, 3))
  
  
    #one-hot encoding
    Y_train = to_categorical(Y_train_orig, num_class)
    Y_test = to_categorical(Y_test_orig, num_class)
    return X_train, Y_train, X_test, Y_test

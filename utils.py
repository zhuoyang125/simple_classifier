from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def get_data():
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
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(X_test, Y_test, model):
    test_scores = model.evaluate(X_test, Y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred=model.predict(X_test)

    import numpy as np
    pred = np.argmax(pred, axis=1)

    label = np.argmax(Y_test,axis=1)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(label, pred)
    return cm


if __name__ == "__main__":
    # TODO Take in a model_path and run some evaluation
    
    pass
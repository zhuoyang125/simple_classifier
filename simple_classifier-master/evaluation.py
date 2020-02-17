from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
import json
import argparse
import os
from matplotlib.image import imread
from utils import get_cifar10
import time 


def evaluate_model(X_test,Y_test,model,config):

    test_batch_size = config['test_batch_size']

    test_scores = model.evaluate(X_test, Y_test, verbose=1, batch_size=test_batch_size)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    label = np.argmax(Y_test,axis=1)

    report = classification_report(label,pred)
    cm = np.array2string(confusion_matrix(label,pred))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = open('results/report-{}.txt'.format(timestr), 'w')
    f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(report,cm))
    f.close()
    
    




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--model_path', help='path directory to model', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--data_path', help='path directory to data folder')
    group.add_argument('-a', '--data_aug', help='initialize data augmentation', action='store_true')

    args = parser.parse_args()

    with open('config/model.json') as f:
        model_config = json.load(f)
    with open('config/training.json') as f:
        train_config = json.load(f)
    
    model_path = args.model_path
    
    model = keras.models.load_model(args.model_path)
    

    if args.data_path:
        train_path = args.data_path + '\\train'
        test_path = args.data_path + '\\test'
        datagen = ImageDataGenerator(rescale = 1./255)
        train_batches = datagen.flow_from_directory(train_path,
                                                    target_size=model_config['0']['input_shape'][:2],
                                                    batch_size=train_config['train_batch_size'],
                                                    class_mode='categorical')
        test_batches = datagen.flow_from_directory(test_path,
                                                target_size=model_config['0']['input_shape'][:2],
                                                batch_size=train_config['test_batch_size'],
                                                class_mode='categorical',
                                                shuffle=False)
        X_test,Y_test = test_batches.next()
        


    else:
        X_train,Y_train,X_test,Y_test = get_cifar10(model_config['0'])
        
    
    #Data Augmentation
    if args.data_aug:
        datagen = ImageDataGenerator(width_shift_range=0.2,
                                horizontal_flip=True,
                                height_shift_range=0.2,
                                rotation_range=20)
        datagen.fit(X_train)
        checkpointer = ModelCheckpoint(filepath='weights/' + model.name + '-{epoch:02d}-{val_loss:.2f}.hdf5',
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', 
                                    period=1)
        model.fit_generator(datagen.flow(X_train,Y_train,batch_size=train_config['train_batch_size']), 
                        steps_per_epoch=None,
                        epochs=train_config['epochs'],
                        validation_data=(X_test,Y_test),
                        callbacks=[checkpointer]
                        )
        evaluate_model(X_test,Y_test,model,train_config)
    
    elif args.data_path:
        checkpointer = ModelCheckpoint(filepath='weights/' + model.name + '-{epoch:02d}-{val_loss:.2f}.hdf5',
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', 
                                    period=1)
        model.fit_generator(train_batches, 
                            steps_per_epoch=None,
                            epochs=train_config['epochs'],
                            validation_data=test_batches,
                            callbacks=[checkpointer]
                            )
        evaluate_model(X_test,Y_test,model,train_config)


    else:
        checkpointer = ModelCheckpoint(filepath='weights/' + model.name + '-{epoch:02d}-{val_loss:.2f}.hdf5',
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', 
                                    period=1)

        model.fit(X_train,Y_train,batch_size=train_config['train_batch_size'], 
                        epochs=train_config['epochs'],
                        validation_split=train_config['train_validation_split'],
                        callbacks=[checkpointer]
                        )
        evaluate_model(X_test,Y_test,model,train_config)






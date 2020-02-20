from tensorflow import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from utils import get_cifar10,get_mnist
from model import get_SimpleNet
from evaluation import evaluate_model
import json
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-a', "--data_aug", help="initialize data augmentation", action='store_true')
parser.add_argument('-s', "--dataset", help="choose 'mnist' or 'cifar10' ")
args = parser.parse_args()

with open('config/model.json') as f:
    model_config = json.load(f)
with open('config/training.json') as f:
    train_config = json.load(f)

if __name__ == "__main__":
    
    model = get_SimpleNet(model_config['0'])

if args.dataset == 'mnist':
    X_train,Y_train,X_test,Y_test = get_mnist(model_config['0'])
else:
    X_train,Y_train,X_test,Y_test = get_cifar10(model_config['0'])

train_batch_size = train_config['train_batch_size']
train_validation_split = train_config['train_validation_split']
epochs = train_config['epochs']

if args.data_aug:
    #Image Data Generator
    datagen = ImageDataGenerator(width_shift_range=0.2,
                            horizontal_flip=True,
                            height_shift_range=0.2,
                            rotation_range=20)
    datagen.fit(X_train)
    checkpointer = ModelCheckpoint(filepath='saved_models/' + model.name + '-{epoch:02d}-{val_loss:.2f}.hdf5',
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', 
                                    period=1)
    model.fit_generator(datagen.flow(X_train,Y_train,batch_size=train_batch_size), 
                    steps_per_epoch=None,
                    epochs=epochs,
                    validation_data=(X_test,Y_test),
                    callbacks=[checkpointer]
                    )
    evaluate_model(X_test,Y_test,model,train_config)

else:
    checkpointer = ModelCheckpoint(filepath='saved_models/' + model.name + '-{epoch:02d}-{val_loss:.2f}.hdf5', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', 
                                    period=1)
    model.fit(X_train, Y_train, batch_size=train_batch_size,epochs=epochs,validation_split=train_validation_split,callbacks=[checkpointer])
    evaluate_model(X_test,Y_test,model,train_config)

    



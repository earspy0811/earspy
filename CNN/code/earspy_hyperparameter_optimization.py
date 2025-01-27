#!/usr/bin/env python
# coding: utf-8

# Import important libraries





import numpy as np
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras import regularizers
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.utils import plot_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import EarlyStopping


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras import regularizers
from keras.utils import plot_model
from kt_utils import *
from cnn_utils import *
from keras.utils.vis_utils import plot_model
import keras.backend as K
import keras
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import progressbar as pb
import dill



import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow





# Load data
def load_data():
    train_dataset = h5py.File('spec_train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_image"][:])
    train_set_y_orig = np.array(train_dataset["train_labels_key"][:])
    train_set_y_orig_1 = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_dataset = h5py.File('spec_test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_image"][:])
    test_set_y_orig = np.array(test_dataset["test_labels_key"][:])
    test_set_y_orig_1 = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    classes = np.array(test_dataset["classes"][:])
    return train_set_x_orig, train_set_y_orig_1, test_set_x_orig, test_set_y_orig_1, classes

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_data()

Y_train_orig=np.int8(Y_train_orig)
Y_test_orig=np.int8(Y_test_orig)

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = convert_to_one_hot(Y_train_orig.astype(int), 4).T
Y_test = convert_to_one_hot(Y_test_orig.astype(int), 4).T





# Define the hyperparameter search space
space = {
    'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'SGD']),
    'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
    'num_filters': hp.choice('num_filters', [32, 64]),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001]),
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'epochs': hp.choice('epochs', [40, 50, 60, 80, 100])
}




# Define a function to create the model
def create_model(args):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(args['num_filters'], (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(args['num_layers']):
        model.add(Conv2D(args['num_filters'], (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(args['dropout_rate']))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(args['dropout_rate']))

    model.add(Dense(4, activation='softmax'))

    # Compile the model
    if args['optimizer'] == 'rmsprop':
        opt = keras.optimizers.rmsprop(learning_rate=args['learning_rate'])
    elif args['optimizer'] == 'adam':
        opt = keras.optimizers.adam(learning_rate=args['learning_rate'])
    elif args['optimizer'] == 'SGD':
        opt = keras.optimizers.sgd(learning_rate=args['learning_rate'])
    else:
        raise ValueError('Invalid optimizer')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    

    return model




# Define a function to train and evaluate the model
def objective(args):
 model = create_model(args)
 early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
 history = model.fit(X_train, Y_train, batch_size=args['batch_size'], epochs=args['epochs'], 
                     validation_split=0.1, callbacks=[early_stopping], verbose=0)
 _, accuracy = model.evaluate(X_test, Y_test, verbose=0)
 return {'loss': -accuracy, 'status': 'ok'}





# Run the hyperparameter search
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=30, trials=trials)





# Print the best hyperparameters
print("Best hyperparameters: ", best)







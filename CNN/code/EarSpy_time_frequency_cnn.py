#!/usr/bin/env python
# coding: utf-8




import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv('9t_rls.csv', header=None, names=['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','q75','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','sharpness','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','maxfreq','class'])



df['labels'] =df['class'].astype('category').cat.codes




df=df.drop(['sharpness'], axis='columns')




# Start preprocess

# Preprocessing Test




#df['smean'] = df['smean']/df['smean'].max()
#df['smean'] = (df['smean'] - df['smean'].min())/(df['smean'].max() - df['smean'].min())
#df['smoothness'] = (df['smoothness'] - df['smoothness'].min())/(df['smoothness'].max() - df['smoothness'].min())
#df['smax'] = (df['smax']-df['smax'].mean())/df['smax'].std()


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from tensorflow import keras





#Some Preprocessing TEST
#clipping test




X = df[['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','q75','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','maxfreq']]
Y = df['labels']
x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, shuffle= True)





print(x_train)




print(x_test.shape)





# The known number of output classes.
num_classes = 10

# Input image dimensions
input_shape = (4,)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)
#x_train_binary = keras.utils.to_categorical(x_train, num_classes)
#x_test_binary = keras.utils.to_categorical(x_test, num_classes)



#x_train = preprocessing.normalize(x_train)
#x_test=preprocessing.normalize(x_test)

x_train = x_train.reshape(x_train.shape[0],25,1)
x_test = x_test.reshape(x_test.shape[0],25,1)





mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
maxi=np.max(x_train,axis=0)
maxiTEST=np.max(x_test,axis=0)
mini=np.min(x_train,axis=0)
miniTEST=np.min(x_test,axis=0)

vmax = 10000
vmin = 10


#Single Feature Scaling
#x_train = x_train/maxi
#x_test = x_test/maxiTEST

#Min Max Normalization

#x_train=(x_train-mini)/(maxi-mini)
#x_test=(x_test-miniTEST)/(maxiTEST-miniTEST)


#z-score normalization
x_train = (x_train - mean)/std
x_test = (x_test - mean)/std


#Clipping

#x_train=x_train.apply(lambda x: vmax if x > vmax else vmin if x < vmin else x)
#x_test=x_test.apply(lambda x: vmax if x > vmax else vmin if x < vmin else x)

#Normalize

# normalize the data attributes


# Check the dataset now 
#x_train[150:160]




print(x_train.shape)





print(np.any(np.isnan(x_train)))





from __future__ import print_function    
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dropout, MaxPooling1D, Activation, BatchNormalization, Dense, Flatten, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam




# Check for NaN values
if np.isnan(x_train).any() or np.isnan(x_test).any():
    print("Input data contains NaN values.")
else:
    print("Input data does not contain NaN values.")

# Check for infinite values
if np.isinf(x_train).any() or np.isinf(x_test).any():
    print("Input data contains infinite values.")
else:
    print("Input data does not contain infinite values.")





# Replace NaN values with 0
x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)




nan_indices_train = np.where(np.isnan(x_train))
if len(nan_indices_train[0]) > 0:
    print("NaN values found in X_train:")
    for i in range(len(nan_indices_train[0])):
        index = (nan_indices_train[0][i], nan_indices_train[1][i], nan_indices_train[2][i])  # Adjust for 3D data
        value = x_train[index]
        print("Index:", index, "Value:", value)

# Find indices of NaN values in X_val
nan_indices_val = np.where(np.isnan(x_test))
if len(nan_indices_val[0]) > 0:
    print("\nNaN values found in X_val:")
    for i in range(len(nan_indices_val[0])):
        index = (nan_indices_val[0][i], nan_indices_val[1][i], nan_indices_val[2][i])  # Adjust for 3D data
        value = x_test[index]
        print("Index:", index, "Value:", value)





print(x_train.shape)





model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(25,1)))  # X_train.shape[1] = No. of Columns

model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv1D(128, 8, padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(2), strides=1))
model.add(Conv1D(64, 8, padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(4), strides=1))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
#model.add(Conv1D(64, 8, padding='same'))
#model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10)) # Target class number
model.add(Activation('softmax'))
model.add(Dropout(0.25))
#opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
opt = keras.optimizers.Adam(lr=0.0001)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#model.summary()





model.compile(loss='categorical_crossentropy', optimizer=opt ,metrics=['accuracy'])
#model.summary()




#print(x_test.shape)
#print(y_test_binary.shape)
model.summary()





#print(y_test)





batch_size = 64
epochs = 80
history=model.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_split=0.1)





predictions = model.predict(x_test)
#print(predictions)
preds = model.evaluate(x = x_test,y = y_test_binary)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# Drawing Confusion matrix


from sklearn.metrics import confusion_matrix



conf_matrix=confusion_matrix(np.argmax(y_test_binary,axis=1),np.argmax(predictions,axis=1))


#confusion_matrix = confusion_matrix(y_test_binary.argmax(axis=1), predictions.argmax(axis=1))
#conf_matrix = tf.math.confusion_matrix(labels=y_test_binary,
                                       #predictions=predictions)
print(conf_matrix)


# Drawing Loss Graphs


plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],color="red")
plt.title('Training Loss vs Validation Loss',weight='bold')
plt.ylabel('Loss',weight='bold')
plt.xlabel('epoch',weight='bold')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# Drawing Training Accuracy vs. vallidation accuarcy


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'],color="red")
plt.title('Training Accuracy Vs. Validation Accuracy',weight='bold')
plt.ylabel('Accuracy',weight='bold')
plt.xlabel('epoch',weight='bold')
plt.legend(['Training', 'validation'], loc='lower right')
plt.show()








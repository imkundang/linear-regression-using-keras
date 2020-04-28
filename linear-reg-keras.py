import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from numpy import genfromtxt
from sklearn.datasets import load_boston


boston = load_boston()
features = np.array(boston.data)
labels = np.array(boston.target)

#split test train  data from train data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,random_state=5,test_size=.3)

ytrain = np.asarray(ytrain)
ytest = np.asarray(ytest)
xtrain=np.asarray(xtrain)
xtest=np.asarray(xtest)


model=Sequential()
model.add(Dense(32,activation='relu',input_dim=xtrain.shape[1]))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

history=model.fit(xtrain,ytrain,epochs=100,batch_size=32,validation_data=(xtest,ytest))

def plot_learningCurve(history, epochs):
    

    # Plot training & validation accuracy
    epoch_range = range(1, epochs+1)
    '''plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()'''
    
    # Plot training & validation loss values\n",
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


    import matplotlib.pyplot as plt
%matplotlib inline
plot_learningCurve(history, 100)
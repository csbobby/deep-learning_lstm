'''
    Author: bobby
    Date created: Feb 1,2016
    Date last modified: May 10, 2016
    Python Version: 2.7
'''
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np


def lstm_training(X_train,y_train,input_dim,nb_classes):
	#settings
	param={}
	param['nb_epoch']=100
	param['batch_size']=64
	param['validation_split']=0.1
	print 'traning',param

	# Here's a Long Short Term Memory (LSTM)
	model = Sequential()
	model.add(LSTM(input_dim = 3561, output_dim = 3000,return_sequences = False))
	model.add(Dense(input_dim = 3000, output_dim = 1))
	model.add(Activation("linear"))
	
	# we'll use MSE (mean squared error) for the loss, and RMSprop as the optimizer
	model.compile(loss='mse', optimizer='rmsprop')
	
	from keras.utils.dot_utils import Grapher
	grapher = Grapher()
	grapher.plot(model, 'model.png')

	#if the evalidation error decreased after one epoch, save the model 
	checkpointer =ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
	# the callback function for logging loss 
	class LossHistory(keras.callbacks.Callback):  
		def on_train_begin(self, logs={}):  
			self.losses = []
		def on_batch_end(self, batch, logs={}):  
			self.losses.append(logs.get('loss'))  
	# define a callback object
	history = LossHistory()  
	print history.losses
	
	print("start train process...")
	hist = model.fit(X_train, y_train,nb_epoch=param['nb_epoch'], batch_size=param['batch_size'], \
	    validation_split=param.get('validation_split'), show_accuracy=True, verbose=0 \
        , callbacks=[checkpointer,history]
        )
	
	loss = hist.history.get('loss')
	val_loss = hist.history.get('val_loss')
	
		
	return model,loss,val_loss
'''
    Author: bobby
    Date created: Nov 11,2015
    Date last modified: Nov 11, 2016
    Python Version: 2.7
'''
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np


def predicting(model,X_test,resultsfname):
	print("Generating test predictions...")
	preds = model.predict_classes(X_test, verbose=0)

	def write_preds(preds, fname):
		pd.DataFrame({"SampleID": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

	write_preds(preds, resultsfname)
	return preds

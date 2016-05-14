'''
    Author: bobby
    Date created: Feb 1,2016
    Date last modified: Apr 10, 2016
    Python Version: 2.7
'''
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import scipy as sp


def testing(X_test,y_test,resultsfname,evaluationfname):
	print("Evaluate the results...")

	preds = model.predict_classes(X_test, verbose=0)
	def write_preds(preds, fname):
		pd.DataFrame({"SampleID": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
	write_preds(preds, resultsfname)
	
	spearmanr_corr = sp.spearmanr(y_test, preds)[0, 1]
	print "Pearson Correlation",p_corr
	
	objective_score = model.evaluate(X_test, y_test, batch_size=32)
	print "objective_score",objective_score

	return spearmanr_corr,objective_score


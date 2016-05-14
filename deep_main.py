'''
    Author: bobby
    Date created: Feb 1,2016
    Date last modified: May 14, 2016
    Python Version: 2.7
'''
import pandas as pd
import numpy as np

from mlp_training import mlp_training
from testing import testing
import sys
sys.path.append('../')
from ws_py import datapreparation as dp
from IterationGraph import *

if __name__ == "__main__":
	print 'start'
	did = 0
	fid = 1
	datasets=['3W','40W','68W','100W'];	datasets=datasets[did]# MSRA:3W 68W VSO_CC:40W 100W
	features=['ALL','USER','DL','VISUAL'];	features=features[fid]
	resulttitle='features'
	sname=features
	model_save_filename = None
	model_save_collection = {}
	
	
	## step 1: load data
	print "step 1: load data..."
	datadir,X_train,y_train,X_test,y_test = dp.feature_loader(datasets,features)
	#results and evaluation fnames
	resultsfname = datadir+'results_'+resulttitle+'_'+datasets+'_'+features+'.txt'
	evaluationfname = datadir+'evaluation_'+resulttitle+'_'+datasets+'_'+features+'.txt'
	input_dim = X_train.shape[1]
	nb_classes = len(np.unique(y_train))
	
	print 'training...'
	model,loss,val_loss = mlp_training(X_train,y_train,input_dim)
	
	history_path = './IterationGraph/%s_history.txt'%features 
	graph_out_path = './IterationGraph/%s_loss.png'%features
	IterationGraph(loss,val_loss,history_path,graph_out_path,features)
	
	print 'testing...'
	spearmanr_corr,objective_score = testing(X_test,y_test,resultsfname,evaluationfname)

	#preds = predicting(model,X_test,resultsfname)



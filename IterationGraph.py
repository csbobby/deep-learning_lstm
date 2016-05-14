'''
    Author: bobby
    Date created: Feb 1,2016
    Date last modified: May 14, 2016
    Python Version: 2.7
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def IterationGraph(loss,val_loss,history_path,graph_out_path,features):
	history_f = open(history_path,'w')
	length_history = len(loss)
	for i in range(length_history):
		history_f.write(str(loss[i]))
		history_f.write('\t')
		history_f.write(str(val_loss[i]))
		history_f.write('\n')
	
    df = pd.read_table(history_path,names=['loss','val_loss'])
    loss_plot = df.plot(title='%s_training'%features)
    loss_fig = loss_plot.get_figure()
    loss_fig.savefig(graph_out_path)
    #loss_fig.savefig('USER_loss.png')




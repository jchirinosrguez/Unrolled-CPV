"""
ISTA_Runfile:
Main file to train or test the unrolled ISTA architecture. 
Example of settings are provided bellow, To proceed please 
change parameters bellow and run the file. 
"""


import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional
import time
from ISTA_Network import Network
torch.cuda.empty_cache()


##################train conditions###################
number_epochs=15000
learning_rate=0.001
train_batch_size= 5
val_batch_size=  5
test_batch_size=1
train_conditions=[number_epochs, learning_rate, train_batch_size, val_batch_size,test_batch_size]
#####################################################

################Path_Example##########################
#Dataset1
#Indicate PATH as suits you
path_train= PATH+"/Dataset1/training"
path_validation=PATH+"/Dataset1/validation"
path_test=PATH+"/Dataset1/test"
H=np.load(PATH+'/Dataset1/'+'H.npy',allow_pickle=True)
H=torch.from_numpy(H).type(torch.cuda.DoubleTensor)
path_save_model=PATH+"/ISTA_Unrolling/Models/"
paths=[path_train, path_validation, path_test, path_save_model]

##########Network_Initilaization###################
num_layers=14
initial_x0="Null_initialization"
Initialization=[num_layers,H,initial_x0]
###################################################


Net=Network(Initialization,train_conditions,paths)
#To train an architecture, uncomment the next line, number of try is the number of folder of training
# Net.train(number_try=1,need_names='yes')

#To test a trained architecture, provide the model path, an example is given bellow
# Net.test(path_set=path_test,path_model="/workspace/code-phd/ISTA_Unrolling/Models/Trained_Model_111/epoch80",need_names='yes')



"""
Runfile:
Main file to train or test the unrolled primal dual architecture. 
Example of settings are provided bellow, To proceed please 
change parameters bellow and run the file. 
"""



import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional
import time
import argparse
from Network import Network
torch.cuda.empty_cache()
####################PATHS####################################
parser = argparse.ArgumentParser()
parser.add_argument('--Dataset', type=str,required=True)
args = parser.parse_args()
if args.Dataset=='Dataset1':
    path_train="./Dataset1/training"
    path_validation="./Dataset1/validation"
    path_test="./Dataset1/test"
    H=np.load('./Dataset1/'+'H.npy',allow_pickle=True)
    
if args.Dataset=='Dataset2': 
    path_train="./Dataset2/training"
    path_validation="./Dataset2/validation"
    path_test="./Dataset2/test"
    H=np.load('./Dataset2/'+'H.npy',allow_pickle=True)
path_save_model="./Primal_Dual_Unrolling/Models/"
H=torch.from_numpy(H).type(torch.cuda.DoubleTensor)
paths=[path_train, path_validation, path_test, path_save_model]




####################Network Initilaization##############
num_layers=22
initial_x0="Null_initialization"
rho=10
Initialization=[num_layers,H,initial_x0,rho]
########################################################



##################Train conditions######################
number_epochs=10000
learning_rate=0.00001
train_batch_size= 5
val_batch_size=  5
test_batch_size=1
train_conditions=[number_epochs, learning_rate, train_batch_size, val_batch_size,test_batch_size,args.Dataset]
########################################################


Net=Network(Initialization,train_conditions,paths)
#To train uncomment next line
# Net.train(number_try=1,need_names='yes')

#To test uncomment next lines
# Net.test(path_set=path_test,path_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/Trained_Model_11/epoch1735",need_names='yes')
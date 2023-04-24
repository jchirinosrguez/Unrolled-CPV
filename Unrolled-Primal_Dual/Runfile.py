import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional
import time
import argparse
from Network import Network

torch.cuda.empty_cache()
torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--Dataset', type=str,required=True)
args = parser.parse_args()
if args.Dataset=='Dataset1':
    path_train="/workspace/code-phd/Dataset1/training"
    path_validation="/workspace/code-phd/Dataset1/validation"
    path_test="/workspace/code-phd/Dataset1/test"
    H=np.load('/workspace/code-phd/Dataset1/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/"
if args.Dataset=='Dataset2': 
    path_train="/workspace/code-phd/Real_Data/Dataset2/training"
    path_validation="/workspace/code-phd/Real_Data/Dataset2/validation"
    path_test="/workspace/code-phd/Real_Data/Dataset2/test"
    H=np.load('/workspace/code-phd/Real_Data/Dataset2/'+'H.npy',allow_pickle=True)
if args.Dataset=='Dataset3':     
    path_train="/workspace/code-phd/Dataset3/training"
    path_validation="/workspace/code-phd/Dataset3/validation"
    path_test="/workspace/code-phd/Dataset3/test"
    H=np.load('/workspace/code-phd/Dataset3/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/"
if args.Dataset=='Dataset4':
    path_train="/workspace/code-phd/Real_Data/Dataset4/training"
    path_validation="/workspace/code-phd/Real_Data/Dataset4/validation"
    path_test="/workspace/code-phd/Real_Data/Dataset4/test"
    H=np.load('/workspace/code-phd/Real_Data/Dataset4/'+'H.npy',allow_pickle=True)
if args.Dataset=='Dataset5':
    path_train="/workspace/code-phd/Real_Data/Dataset5/training"
    path_validation="/workspace/code-phd/Real_Data/Dataset5/validation"
    path_test="/workspace/code-phd/Real_Data/Dataset5/test"
    H=np.load('/workspace/code-phd/Real_Data/Dataset5/'+'H.npy',allow_pickle=True)

if args.Dataset=='Dataset6':
    path_train="/workspace/code-phd/Real_Data/Dataset6/training"
    path_validation="/workspace/code-phd/Real_Data/Dataset6/validation"
    path_test="/workspace/code-phd/Real_Data/Dataset6/test"
    H=np.load('/workspace/code-phd/Real_Data/Dataset6/'+'H.npy',allow_pickle=True)
    
if args.Dataset=='Dataset7':
    path_train="/workspace/code-phd/Real_Data/Dataset7/training"
    path_validation="/workspace/code-phd/Real_Data/Dataset7/validation"
    path_test="/workspace/code-phd/Real_Data/Dataset7/test"
    H=np.load('/workspace/code-phd/Real_Data/Dataset7/'+'H.npy',allow_pickle=True)


    
if args.Dataset=='IFPEN_Data0':
    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset0/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset0/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset0/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset0/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data0/"

if args.Dataset=='IFPEN_Data1':

    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset1/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset1/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset1/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset1/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data1/"
    
if args.Dataset=='IFPEN_Data2':

    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset2/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset2/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset2/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset2/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data2/"
    
if args.Dataset=='IFPEN_Data3':

    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset3/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset3/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset3/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset3/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data3/"
    
if args.Dataset=='IFPEN_Data4':
    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset4/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset4/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset4/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset4/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data4/"

if args.Dataset=='IFPEN_Data5':
    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset5/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset5/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset5/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset5/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data5/"
    
if args.Dataset=='IFPEN_Data6':
    path_train='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset6/training'
    path_validation='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset6/validation'
    path_test='/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset6/test'
    H=np.load('/workspace/code-phd/IFPEN_Data/Simulated_Data/Dataset6/'+'H.npy',allow_pickle=True)
    path_save_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/IFPEN_Data6/"





    

H=torch.from_numpy(H).type(torch.cuda.DoubleTensor)





paths=[path_train, path_validation, path_test, path_save_model]




#Initilaization
num_layers=22
initial_x0="Null_initialization"
rho=10
# initial_x0="Wiener_initialization"
Initialization=[num_layers,H,initial_x0,rho]




#train conditions
number_epochs=10000
learning_rate=0.00001
train_batch_size= 5
val_batch_size=  5
test_batch_size=1
train_conditions=[number_epochs, learning_rate, train_batch_size, val_batch_size,test_batch_size,args.Dataset]
Net=Network(Initialization,train_conditions,paths)
# Net.train(number_try=1,need_names='yes')

# Net.plot_signals("/workspace/code-phd/Dataset1/test/Groundtruth/x_Gr_te_0.npy","/workspace/code-phd/Primal_Dual_Unrolling/Models/Trained_Model_11/epoch1735-22layers")
# Net.plot_stepsizes("/workspace/code-phd/Primal_Dual_Unrolling/Models/Trained_Model_999/epoch146")
# Net.test(path_set=path_test,path_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/Trained_Model_999/epoch146",need_names='yes')
# Net.test(path_set=path_test,path_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/Trained_Model_33/epoch2013-16layers",need_names='yes')

Net.test(path_set=path_test,path_model="/workspace/code-phd/Primal_Dual_Unrolling/Models/Trained_Model_11/epoch1735-22layers",need_names='yes')
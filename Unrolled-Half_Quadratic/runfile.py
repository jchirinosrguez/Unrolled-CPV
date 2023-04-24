import time
import torch.nn as nn  
import numpy as np
import torch
from network import Net_class
import cProfile
import argparse
import os
torch.cuda.empty_cache()



parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--function', type=str, required=True)
parser.add_argument('--number_layers', type=int,required=True)
parser.add_argument('--Initialization', type=str,required=True)
parser.add_argument('--penalization_cvx', type=str,required=True)
parser.add_argument('--penalization_ncvx', type=str,required=True)
parser.add_argument('--path', type=str,required=True)
parser.add_argument('--lamda_cvx', type=str)
parser.add_argument('--lamda_ncvx', type=str)
parser.add_argument('--delta_cvx', type=str)
parser.add_argument('--delta_ncvx', type=str)
parser.add_argument('--architecture_lambda', type=str)




args = parser.parse_args()
if args.mode in ["learning_lambda_MM","MM","3MG"]:
    mode= args.mode
else:
    print("Give valid mode!")
if args.function in ['train', 'test']:
    function=args.function
else:
    print('Give valid function to execute!')
if args.number_layers:
    number_layers=args.number_layers
if args.Initialization in ["Wiener_initialization","Null_initialization"]:
    intial_x0=args.Initialization
else:
    print("Give valid initialization!")

if args.penalization_cvx in ['fair','green','GMc','welsh','cauchy']:
    number_penalization_cvx=args.penalization_cvx
    
else:
    print('Give valid penalization!')

if args.penalization_ncvx:
    number_penalization_ncvx=args.penalization_ncvx
    
if args.path in ['Dataset1','Dataset2']:
    Path_name=args.path
else:
    print('Please give a path of a valid Dataset')

if args.lamda_cvx:
    lamda_cvx=float(args.lamda_cvx)
if args.lamda_ncvx:
    lamda_ncvx=float(args.lamda_ncvx)    
if args.delta_cvx:
    delta_s_cvx=float(args.delta_cvx)
if args.delta_ncvx:
    delta_s_ncvx=float(args.delta_ncvx)

if args.architecture_lambda:
    architecture_lambda=args.architecture_lambda

Mk=5
if Path_name=='Dataset1':
    Path='./Dataset1/'
    H=np.load('./Dataset1/'+'H.npy',allow_pickle=True)
if Path_name=='Dataset2':
    Path='./Dataset2/'
    H=np.load('./Dataset2/'+'H.npy',allow_pickle=True)


n_in= np.shape(H)[1]
n_out=n_in
p= np.shape(H)[0]
    
    
H=torch.from_numpy(H).type(torch.cuda.FloatTensor)
Ht=torch.transpose(H,0,1)
V1=torch.mm(Ht,H)
V2=torch.eye(n_in).type(torch.cuda.FloatTensor)# L=Id for now
V3=torch.cat((V2,lamda_cvx*V2,lamda_ncvx*V2), 1).type(torch.cuda.FloatTensor) # [Id| lamda*Lt], L=Id






Initializations=[V1,V2,V3,H,Ht,n_in,n_out,Mk,p,number_layers,lamda_cvx,lamda_ncvx,delta_s_cvx,delta_s_ncvx,intial_x0]

number_epochs=1000
train_batch_size= 10
val_batch_size=  10
train_conditions=[number_epochs,train_batch_size,val_batch_size,architecture_lambda]
Folders=[Path+'training/',Path+'validation/',Path+'test/',Path]
test_batch_size=1
test_conditions=[test_batch_size]



Network = Net_class(Initializations, train_conditions, test_conditions, number_penalization_cvx,number_penalization_ncvx, Folders, mode)

        
if function=='train':
    Network.train(lr=0.0001,need_names='yes',number_try=500)
    
if function=='test':
    if Path_name== 'Dataset1':
        Network.test(path_set='./Dataset1/test',path_model='./Dataset1/Trained_Model_learning_lambda_MM_500/epoch75', need_names="yes")
    if Path_name =="Dataset2":
        Network.test(path_set='./Dataset2/test',path_model='./Dataset2/Trained_Model_learning_lambda_MM_500/epoch80',need_names="no")
    
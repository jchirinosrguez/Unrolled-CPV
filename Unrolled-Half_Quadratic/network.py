"""
Network.
Class
-----------
Net_class : Includes the main training and testing of the Unrolled Half-Quadratic architecture.
"""



from __future__ import print_function
import torch
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from model import myModel
from tools import F, SNR,MyMSE, PSNR,TSNR
import cProfile
import torch.nn.functional as FF
from modules import MyDataset

class Net_class(nn.Module):

    """""
    Creates an instance of the model, loads data and performs training and testing.

    Attributes
    ----------
    number_layers (float) : Number of layers of the network
    V1 (torch.FloatTensor): convolution filter corresponding to Ht*H
    V2 (torch.FloatTensor): convolution filter corresponding to L
    V3 (torch.FloatTensor): convolution filter corresponding to [Id|lambda*Lt]
    n_in    (float)       : size of input tensor to layers 1 and 2,Bkt | 2*n_in for 3
    n_out   (float)       : size of output tensor to layers 1, 2, 3 and Bk
    Mk      (float)       : Number of columns in Bk
    lamda   (float)       : Regularization term
    path_train (string)   : path to training folder
    path_test (string)    : path to testing folder
    path_save (string)    : path to save matrix bk and trained model
    Ht (torch.FloatTensor): Ht
    model (Mymodel)       : the model
    dtype...............
    loss_fun...........
    learning rate
    momentum
    number of epochs
    """""

    def __init__(self, initializations, train_conditions, test_conditions, number_penalization,number_penalization1, folders, mode):
        super(Net_class, self).__init__()
        self.V1, self.V2, self.V3, self.H, self.Ht, self.n_in, self.n_out, self.Mk, self.p, self.number_layers, self.lamda,self.lamda1, self.delta_s,self.delta_s1, self.initial_x0 = initializations

        self.number_epochs, self.train_batch_size, self.val_batch_size,self.architecture_lambda = train_conditions
        self.test_batch_size = test_conditions[0]
        self.path_train, self.path_validation, self.path_test, self.path_save = folders
       
        self.mode = mode
        self.number_penalization = number_penalization
        self.number_penalization1 = number_penalization1
        self.dtype=torch.cuda.FloatTensor
        self.model = myModel(self.number_layers, self.V1, self.V2, self.V3, self.H, self.Ht, self.n_in, self.n_out,
 self.p, self.lamda,self.lamda1, self.delta_s,self.delta_s1, self.Mk, self.dtype, self.mode, 
                             self.number_penalization,self.number_penalization1,self.architecture_lambda).cuda()
        self.loss_fun= nn.MSELoss()

    def CreateLoader(self, need_names,path_set=None):
        if path_set is not None:
            without_extra = os.path.normpath(path_set)
            last_part = os.path.basename(without_extra)
            if last_part == "training" :
                train_data = MyDataset(self.path_train,self.initial_x0, need_names)
                self.loader = DataLoader(train_data, batch_size=1, shuffle=False)
            if last_part == "validation":
                val_data = MyDataset(self.path_validation,self.initial_x0, need_names)
                self.loader = DataLoader(val_data, batch_size=1, shuffle=False)
            if last_part == "test":
                test_data = MyDataset(self.path_test,self.initial_x0, need_names)
                self.loader = DataLoader(test_data, batch_size=1, shuffle=False)
            self.size = len(self.loader)
        else:
          
            # For training purposes
            train_data = MyDataset(self.path_train,self.initial_x0, need_names)
            self.train_loader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)
            val_data = MyDataset(self.path_validation,self.initial_x0, need_names)
            self.val_loader = DataLoader(val_data, batch_size=self.val_batch_size, shuffle=True)





   
    def train(self,lr,need_names,path_model=None,number_try=None):
      
        """
        Trains unfolded MM to learn regularization. 
        Parameters
        ----------
            lr    (int): learning rate for training
            block (int): number of the layer to be trained, numbering starts at 0 (default is 0)
            path_model : path for saved model to resume training.
            number_try : number of try of a certain training.
        """

        if  self.mode == "learning_lambda_MM" or self.mode == "learning_lambda_3MG" or self.mode =="Deep_equilibrium" or self.mode =="Deep_equilibrium_3MG" or self.mode=="learning_Bk" or self.mode=="learning_total":

            epoc=0
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            if path_model is not None:
                checkpoint = torch.load(path_model)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoc = checkpoint['epoch'] +1
            else:
                if not os.path.exists(os.path.join(self.path_save,'Trained_Model_'+self.mode+'_'+str(number_try))):

                    os.makedirs(os.path.join(self.path_save,'Trained_Model_'+self.mode+'_'+str(number_try)))
                file_object = open(os.path.join(self.path_save,'Trained_Model_'+self.mode+'_'+str(number_try)) + "/readme.txt", "a")
                file_object.writelines(["Mode: " + self.mode + '\n',
                                        "Optimizer: " + str(optimizer) + '\n',
                                        "learning_rate:"+ str(lr)+ '\n',
                                        "Number layers: " + str(self.number_layers) + '\n',
                                        "Penalization_number:" + str(self.number_penalization)+ '\n',
                                        "Delta: " + str(self.delta_s) + '\n',
                                        "Initial_lambda: " + str(self.lamda) + '\n',
                                        "Initial_x0: " + str(self.initial_x0) + '\n',
                                         "batch_val_size: " + str(self.val_batch_size) + '\n',
                                        "batch_train_size: " + str(self.train_batch_size) + '\n',
                                        "Architecture_lambda: " + str(self.architecture_lambda) + '\n',
                                       ])
                file_object.close()
            

            self.CreateLoader(need_names=need_names)
            loss_epochs = []
            val_loss_epochs = []    
            for epoch in range(self.number_epochs):
                print("epoch is",epoch)
                if epoch+epoc==0 :
                    self.model.Layers.eval()
                if epoch+epoc >0 :
                    self.model.Layers.train()
                running_loss = 0.0
                total_SNR=0
                for i, minibatch in enumerate(self.train_loader, 0):
                    if self.initial_x0=="Wiener_initialization":
                        if need_names=="yes":
                            [name,x_true, x_degraded,x0] = minibatch
                        
                        if need_names=="no":
                            [x_true, x_degraded,x0] = minibatch
                        
                    
                        x_true = Variable((x_true).type(self.dtype), requires_grad=False)
                        x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                        x0 = Variable((x0).type(self.dtype), requires_grad=False)
                    if self.initial_x0=="Null_initialization":
                        if need_names=="yes":
                            [name,x_true, x_degraded] = minibatch
                      
                        if need_names=="no":
                            [x_true, x_degraded] = minibatch
                        
                        x_true = Variable((x_true).type(self.dtype), requires_grad=False)
                        x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                        x0=Variable(torch.zeros((self.train_batch_size,x_true.size()[1])).type(self.dtype), requires_grad=False)
                       
                        
                 
                    Ht_x_degraded = torch.transpose(torch.mm(self.Ht, torch.transpose(x_degraded, 0, 1)).detach(), 0, 1)
                    if epoch+epoc==0:
                        
                        x_pred = self.model(x0, x_degraded, x_true, Ht_x_degraded, self.mode,False).detach()
                        
                        loss=self.loss_fun(x_pred, x_true)
                        snr=SNR(x_true,x_pred).detach()
                        running_loss += loss.item()
                        total_SNR+=snr
                    if epoch+epoc >0:
                  
                        optimizer.zero_grad()
                        x_pred = self.model(x0, x_degraded, x_true, Ht_x_degraded, self.mode,False)
                        loss=self.loss_fun(x_pred, x_true)
                        loss.backward()     
                        optimizer.step()
                        snr=SNR(x_true,x_pred).detach()
                        running_loss += loss.item()
                        total_SNR+=snr
                        torch.autograd.set_detect_anomaly(True)
                        
                loss_epochs.append(running_loss / (len(self.train_loader)))
                print("train loss for epoch", epoch + epoc , "is", running_loss / (len(self.train_loader)), "And AVG SNR is", total_SNR/(len(self.train_loader)))


                #saving model here
                torch.save({
                    'epoch': epoch + epoc ,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss, },
                    os.path.join(self.path_save,'Trained_Model_'+self.mode+'_'+str(number_try))+ '/epoch'+ str(epoch + epoc ))
                
                #Evaluation on validation set
                self.model.eval()
                with torch.no_grad():
                    loss_current_val = 0
                    total_SNR_val=0
                    for minibatch in self.val_loader:
                        if self.initial_x0=="Wiener_initialization":
                            if need_names=="yes":
                                [name,x_true, x_degraded, x0] = minibatch
                            if need_names=="no":
                                [x_true, x_degraded, x0] = minibatch
                        
                            x_true = Variable((x_true).type(self.dtype), requires_grad=False)
                            x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                            x0 = Variable((x0).type(self.dtype), requires_grad=False)
                        if self.initial_x0=="Null_initialization":
                            if need_names=="yes":
                                [name,x_true, x_degraded] = minibatch
                            if need_names=="no":
                                [x_true, x_degraded] = minibatch
                            

                            x_true = Variable((x_true).type(self.dtype), requires_grad=False)
                            x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                            x0=Variable(torch.zeros((self.val_batch_size,x_true.size()[1])).type(self.dtype), requires_grad=False)
                        
                        Ht_x_degraded = torch.transpose(torch.mm(self.Ht, torch.transpose(x_degraded, 0, 1)), 0, 1)
                        x_pred = self.model(x0, x_degraded, x_true, Ht_x_degraded, self.mode,False) 
                        loss_val = self.loss_fun(x_pred, x_true)
                        snr_val=SNR(x_true,x_pred)
                        loss_current_val += torch.Tensor.item(loss_val)
                        total_SNR_val+= snr_val
                    val_loss_epochs.append(loss_current_val / (len(self.val_loader)))
                    print("val loss for epoch", epoch + epoc , "is", loss_current_val / (len(self.val_loader))," And SNR_val is:", total_SNR_val/(len(self.val_loader)))
                    #saving learning evolution in readme file
                    file_object = open(
                        os.path.join(self.path_save, 'Trained_Model_' + self.mode + '_' + str(number_try)) + "/readme.txt",
                        "a")
                    file_object.writelines([ "Train loss for epoch "+ str(epoch + epoc)+ "is: "+ str( running_loss / (len(self.train_loader)))+'\n', "Train SNR for epoch "+ str(epoch + epoc)+ "is: "+ str(  total_SNR / (len(self.train_loader)))+'\n',
                                            "Val loss for epoch" +str(epoch + epoc ) +"is: "+ str(loss_current_val / (len(self.val_loader)))+ '\n', "Val SNR for epoch" +str(epoch + epoc) +"is: "+ str(total_SNR_val / (len(self.val_loader)))+ '\n'
                                           ])
                    file_object.close()
                    #plotting learning curves
                    plt.plot(loss_epochs,color='black', linestyle='dashed', linewidth=1)
                    plt.plot(val_loss_epochs,color='blue' ,linestyle='dashed', linewidth=1)
                    axes = plt.gca()
                    axes.set_xlabel('Epochs')

                    axes.set_ylabel('Average MSE loss')
                    plt.legend(["Training loss","Validation loss"])
                    plt.savefig(os.path.join(self.path_save, 'Trained_Model_' + self.mode + '_' + str(number_try))+'/Trained_Model_' + self.mode + '_' + str(number_try)+"Training_curves.png")
            return
            
    def test(self, path_set=None,path_model=None, need_names="no", path_signal=None,save_estimate=False):
        
        if self.mode == "learning_lambda_3MG" or self.mode == "learning_lambda_MM" or self.mode == 'Deep_equilibrium' or self.mode == 'Deep_equilibrium_3MG' :
            checkpoint = torch.load(path_model,map_location='cuda:0')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
                         
            
        if path_signal is not None :
            name=os.path.split(path_signal)[1]
            x_true=torch.unsqueeze(torch.tensor(np.load(path_signal, allow_pickle=True)),0)
            x_degraded=torch.unsqueeze(torch.tensor(np.load(path_signal.replace('Groundtruth','Degraded').replace('_Gr_','_De_'), allow_pickle=True)),0)
            x_true = Variable(x_true.type(self.dtype), requires_grad=False)
            x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)

            if self.initial_x0=="Null_initialization":
                x_0=Variable(torch.zeros((1,x_true.size()[1])).type(self.dtype), requires_grad=False)
            Ht_x_degraded = torch.transpose(torch.mm(self.Ht, torch.transpose(x_degraded, 0, 1)), 0, 1).detach()
            t0=time.time()
            
            x_pred = self.model(x_0, x_degraded, x_true, Ht_x_degraded, self.mode,False).detach()
            t1=time.time()
           
            
            loss = self.loss_fun(x_pred, x_true).detach()
            snr=SNR(x_true,x_pred)
            tsnr=TSNR(x_true,x_pred)
            print("loss is:",loss, "and SNR is", snr,  "and TSNR is", tsnr)
            print("Execution time is",t1-t0)
            return x_pred
         
          
                


        else:
            with torch.no_grad():
                self.CreateLoader(need_names,path_set)
                total_loss = 0
                total_time =0
                high_MSE=0
                total_SNR=0
                total_tSNR=0
                i=0
                print(self.size)
                MSE_list=[]
                SNR_list=[]
                TSNR_list=[]
                for minibatch in self.loader:

                    if self.initial_x0=="Null_initialization":

                        if need_names =="yes":
                            name, x_true, x_degraded = minibatch
                        

                        if need_names =="no":
                            x_true, x_degraded = minibatch
                        x_true = Variable(x_true.type(self.dtype), requires_grad=False)
                        x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                        x_0=Variable(torch.zeros((self.test_batch_size,x_true.size()[1])).type(self.dtype), requires_grad=False)

                    if self.initial_x0=="Wiener_initialization":

                        if need_names =="yes":
                            name, x_true, x_degraded,x_0 = minibatch


                        if need_names =="no":
                            x_true, x_degraded,x_0 = minibatch
                        x_true = Variable(x_true.type(self.dtype), requires_grad=False)
                        x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                        x_0 = Variable((x_0).type(self.dtype), requires_grad=False)

                    Ht_x_degraded = torch.transpose(torch.mm(self.Ht, torch.transpose(x_degraded, 0, 1)), 0, 1)
                  
                    t0=time.time()
                    x_pred = self.model(x_0, x_degraded, x_true, Ht_x_degraded, self.mode,False)
                    t1=time.time()
                    loss = (self.loss_fun(x_true, x_pred).detach())
                    snr=SNR(x_true,x_pred).detach()
                    tsnr=TSNR(x_true,x_pred).detach()
                    total_loss += loss
                    total_time += t1-t0
                    total_SNR+= snr
                    total_tSNR +=tsnr
                    
                    MSE_list.append(loss)
                    SNR_list.append(snr)
                    TSNR_list.append(tsnr)
                    i+=1
                #compute metrics STD
                mse_std=0
                for l in MSE_list:
                    mse_std=mse_std+((l-total_loss/self.size)**2)
                mse_std=torch.sqrt(mse_std/(self.size-1))
                
                snr_std=0
                for l in SNR_list:
                    snr_std=snr_std+((l-total_SNR/self.size)**2)
                snr_std=torch.sqrt(snr_std/(self.size-1))
                
                tsnr_std=0
                for l in TSNR_list:
                    tsnr_std=tsnr_std+((l-total_tSNR/self.size)**2)
                tsnr_std=torch.sqrt(tsnr_std/(self.size-1))
                
                print("Average MSE loss is ", float(total_loss / self.size), "Average SNR is ", float(total_SNR / self.size) , 
                      "Avearge TSNR is: ",float(total_tSNR/self.size),"Average execution time is ", float(total_time / self.size))
                print("the standard deviation of MSE loss is", float(mse_std))
                print("the standard deviation of SNR is", float(snr_std))
                print("the standard deviation of TSNR is", float(tsnr_std))
                return  total_loss / self.size

    
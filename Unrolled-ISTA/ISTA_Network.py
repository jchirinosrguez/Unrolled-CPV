"""
ISTA_Network.
Class
-----------
Network : Includes the main training and testing of the Unrolled ISTA.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from module import MyDataset
from ISTA_model import ISTA_model
from ISTA_utils import SNR, TSNR
import torch.nn.functional as FF
Loss_fun= nn.MSELoss()

class Network(nn.Module):
    def __init__(self,Initialization,train_conditions,paths):
        super(Network,self).__init__()
        self.number_layers,self.H,self.initial_x0=Initialization 
        self.number_epochs, self.lr, self.train_batch_size, self.val_batch_size,self.test_batch_size = train_conditions
        self.path_train, self.path_validation, self.path_test, self.path_save_model = paths
        self.model = ISTA_model(self.number_layers,self.H).cuda()
        self.dtype=torch.cuda.DoubleTensor
        
        
        
    def CreateLoader(self,need_names):
        train_data = MyDataset(self.path_train,self.initial_x0, need_names)
        self.train_loader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)
        val_data = MyDataset(self.path_validation,self.initial_x0, need_names)
        self.val_loader = DataLoader(val_data, batch_size=self.val_batch_size, shuffle=True)
        test_data=MyDataset(self.path_test,self.initial_x0, need_names)
        self.test_loader = DataLoader(test_data, batch_size=self.test_batch_size, shuffle=True)
    
    def train(self,number_try,need_names,path_model=None):
        epoc=0
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if path_model is not None:
            checkpoint = torch.load(path_model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoc = checkpoint['epoch'] +1
        else:
            if not os.path.exists(os.path.join(self.path_save_model,'Trained_Model'+'_'+str(number_try))):
                os.makedirs(os.path.join(self.path_save_model,'Trained_Model'+'_'+str(number_try)))
            file_object = open(os.path.join(self.path_save_model,'Trained_Model'+'_'+str(number_try)) + "/readme.txt", "a")
            file_object.writelines(["Optimizer: " + str(optimizer) + '\n',
                                        "learning_rate:"+ str(self.lr)+ '\n',
                                        "Number layers: " + str(self.number_layers) + '\n',
                                         "batch_val_size: " + str(self.val_batch_size) + '\n',
                                        "batch_train_size: " + str(self.train_batch_size) + '\n',
                                       ])
            file_object.close()
            

        self.CreateLoader(need_names=need_names)

        loss_epochs = []
        val_loss_epochs = []    

        for epoch in range(self.number_epochs):
            print("epoch is",epoch)
            if epoch+epoc==0 :
                self.model.eval()
            if epoch+epoc >0 :
                self.model.train()
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
                    
                    
                if epoch+epoc==0:
                    x_pred=self.model(self.H,x0,x_degraded,x_true)
                    loss=Loss_fun(x_pred, x_true).detach()
                    snr=SNR(x_true,x_pred).detach()
                    running_loss += loss.item()
                    total_SNR+=snr
                if epoch+epoc >0:
                    optimizer.zero_grad()
                    x_pred=self.model(self.H,x0,x_degraded,x_true)
                    snr=SNR(x_true,x_pred).detach()
                    loss=Loss_fun(x_pred, x_true)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    total_SNR+=snr
                    torch.autograd.set_detect_anomaly(True)
            loss_epochs.append(running_loss / (len(self.train_loader)))
            print("train loss for epoch", epoch + epoc , "is", running_loss / (len(self.train_loader)), "And AVG SNR is", total_SNR/(len(self.train_loader)))


            # saving model here
            torch.save({
                    'epoch': epoch + epoc ,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss, },
                    os.path.join(self.path_save_model,'Trained_Model'+'_'+str(number_try))+ '/epoch'+ str(epoch + epoc ))
                
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
                            [name,x_true, x_degraded, x0] = minibatch
                            
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

                    x_pred=self.model(self.H,x0,x_degraded,x_true)
                    loss_val=Loss_fun(x_pred, x_true)
                    snr_val=SNR(x_true,x_pred)
                    loss_current_val += torch.Tensor.item(loss_val)
                    total_SNR_val+= snr_val
                val_loss_epochs.append(loss_current_val / (len(self.val_loader)))
                print("val loss for epoch", epoch + epoc , "is", loss_current_val / (len(self.val_loader))," And SNR_val is:", total_SNR_val/(len(self.val_loader)))
                #Saving training evolution in readme file                    
                file_object = open(
                    os.path.join(self.path_save_model, 'Trained_Model'  + '_' + str(number_try)) + "/readme.txt",
                        "a")
                file_object.writelines([ "Train loss for epoch "+ str(epoch + epoc)+ "is: "+ str( running_loss / (len(self.train_loader)))+'\n', "Train SNR for epoch "+ str(epoch + epoc)+ "is: "+ str(  total_SNR / (len(self.train_loader)))+'\n',
                                            "Val loss for epoch" +str(epoch + epoc ) +"is: "+ str(loss_current_val / (len(self.val_loader)))+ '\n', "Val SNR for epoch" +str(epoch + epoc) +"is: "+ str(total_SNR_val / (len(self.val_loader)))+ '\n'
                                           ])
                file_object.close()
                plt.plot(loss_epochs,color='black', linestyle='dashed', linewidth=1)
                plt.plot(val_loss_epochs,color='blue' ,linestyle='dashed', linewidth=1)
                axes = plt.gca()
                axes.set_xlabel('Epochs')
                axes.set_ylabel('Average MSE loss')
                plt.legend(["Training loss","Validation loss"])
                plt.savefig(os.path.join(self.path_save_model, 'Trained_Model'  + '_' + str(number_try))+'/Trained_Model'  + str(number_try)+"Training_curves.png")
        return
    
    
    
    
    
    
    
    def test(self, path_set=None,path_model=None, need_names="no", path_signal=None,save_estimate=False):
    
        checkpoint = torch.load(path_model,map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
            
                              
            
        if path_signal is not None :
            # test method on one given signal
            name=os.path.split(path_signal)[1]
            x_true=torch.unsqueeze(torch.tensor(np.load(path_signal, allow_pickle=True)),0)
            x_degraded=torch.unsqueeze(torch.tensor(np.load(path_signal.replace('Groundtruth','Degraded').replace('_Gr_','_De_'), allow_pickle=True)),0)
            x_true = Variable(x_true.type(self.dtype), requires_grad=False)
            x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
            if self.initial_x0=="Null_initialization":
                x_0=Variable(torch.zeros((1,x_true.size()[1])).type(self.dtype), requires_grad=False)
               
            t0=time.time()
            x_pred=self.model(self.H,x_0,x_degraded,x_true)
            t1=time.time()
            loss = Loss_fun(x_pred, x_true).detach()
            snr=SNR(x_true,x_pred)
            print("loss is:",loss, "and SNR is", snr)
            print("Execution time is",t1-t0)
            return x_pred.detach()

        else:
            #test method on test set
            self.CreateLoader(need_names)
            total_loss = 0
            total_time =0
            high_MSE=0
            total_SNR=0
            total_tSNR=0
            i=0
            MSE_list=[]
            SNR_list=[]
            TSNR_list=[]
            for minibatch in self.test_loader:
                if self.initial_x0=="Wiener_initialization":
                    if need_names =="yes":
                        name, x_true, x_degraded, x_0 = minibatch
                        name=name[0]
                    if need_names =="no":
                        [x_true, x_degraded, x_0] = minibatch
                    x_true = Variable(x_true.type(self.dtype), requires_grad=False)
                    x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                    x_0 = Variable((x_0).type(self.dtype), requires_grad=False)
                if self.initial_x0=="Null_initialization":
                    if need_names =="yes":
                        name, x_true, x_degraded = minibatch
                        name=name[0]
                    if need_names =="no":
                        [x_true, x_degraded] = minibatch
                    x_true = Variable(x_true.type(self.dtype), requires_grad=False)
                    x_degraded = Variable((x_degraded).type(self.dtype), requires_grad=False)
                    x_0=Variable(torch.zeros((self.test_batch_size,x_true.size()[1])).type(self.dtype), requires_grad=False)
             
                t0=time.time()
                x_pred=self.model(self.H,x_0,x_degraded,x_true)
                t1=time.time()
                loss = (Loss_fun(x_true, x_pred).detach())
                snr=SNR(x_true,x_pred).detach()
                tsnr=TSNR(x_true,x_pred).detach()
                MSE_list.append(loss)
                SNR_list.append(snr)
                TSNR_list.append(tsnr)
                total_loss += loss
                total_time += t1-t0
                total_SNR+= snr
                total_tSNR +=tsnr
                i+=1
            
            
            #compute metrics STD
            mse_std=0
            for l in MSE_list:
                mse_std=mse_std+((l-total_loss/len(self.test_loader))**2)
            mse_std=torch.sqrt(mse_std/(len(self.test_loader)-1))
                
            snr_std=0
            for l in SNR_list:
                snr_std=snr_std+((l-total_SNR/len(self.test_loader))**2)
            snr_std=torch.sqrt(snr_std/(len(self.test_loader)-1))
                
            tsnr_std=0
            for l in TSNR_list:
                tsnr_std=tsnr_std+((l-total_tSNR/len(self.test_loader))**2)
            tsnr_std=torch.sqrt(tsnr_std/(len(self.test_loader)-1))
            print("Average MSE loss is ", total_loss / len(self.test_loader), "Average SNR is ", total_SNR / len(self.test_loader) , "Avearge TSNR is: ",float(total_tSNR/len(self.test_loader)),"Average execution time is ", total_time /len(self.test_loader))
            print("the standard deviation of MSE loss is", float(mse_std))
            print("the standard deviation of SNR is", float(snr_std))
            print("the standard deviation of TSNR is", float(tsnr_std))
            return  
    
    
    
    
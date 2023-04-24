from utils import activation_primal, activation_dual
from stepsize_architecture import stepsize_arch
import torch
import torch.nn as nn
Loss_fun= nn.MSELoss()
class layer(torch.nn.Module):
    def __init__(self):
        super(layer, self).__init__()
        self.architecture=stepsize_arch()
    
    def forward(self,H,rho,p,p_old,d,d_old,y):
       
        tau,sigma,rho=self.architecture()
        bp=-tau*torch.mm(H.T,(2*d-d_old).T).T
        p=activation_primal(p+bp,tau)
        
#         bd=sigma*(torch.mm(H,(p).T).T -y)
        bd=sigma*torch.mm(H,(p).T).T 
        d=activation_dual(d+bd,y,sigma,rho)
        
        return p,d
        
class PD_model(torch.nn.Module):
    def __init__(self,num_layers,H,rho):
        super(PD_model,self).__init__()
        self.rho=rho
        self.Layers = nn.ModuleList()
        for i in range(num_layers):
            self.Layers.append(layer())
            
    def forward(self,H,p0,p0_old,d0,d0_old,y,x_true):
        for i, l in enumerate(self.Layers):
            mse=Loss_fun(p0,x_true)
            p,d=self.Layers[i](H,self.rho,p0,p0_old,d0,d0_old,y)
            p0_old=p0
            d0_old=d0
            d0=d
            p0=p
        return p,d
            
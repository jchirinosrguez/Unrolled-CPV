"""
ISTA_model.
Classes
-----------
layer:       One layer in the unrolled ISTA architecture.
ISTA_model : Unrolled ISTA model.
"""


from ISTA_utils import shrink
import torch
import torch.nn as nn
from ISTA_joint_architecture import stepsize_regularization_arch
Loss_fun= nn.MSELoss()


class layer(torch.nn.Module):
    def __init__(self):
        super(layer, self).__init__()
        self.architecture=stepsize_regularization_arch()
    
    def forward(self,H,x,y):
        gamma,xsi=self.architecture()
        x=shrink(x-gamma*torch.mm(H.T,(torch.mm(H,x.T).T -y).T).T,gamma*xsi)
        return x
        
class ISTA_model(torch.nn.Module):
    def __init__(self,num_layers,H):
        super(ISTA_model,self).__init__()
        self.Layers = nn.ModuleList()
        for i in range(num_layers):
            self.Layers.append(layer())
            
    def forward(self,H,x0,y,x_true):
        for i, l in enumerate(self.Layers):
            x=self.Layers[i](H,x0,y)
            x0=x
        return x
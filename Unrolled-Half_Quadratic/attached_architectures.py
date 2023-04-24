"""
attached_architectures.
Class
-----------
R_Arch : Different possible architectures to learn regularization hyperparameters, .
"""
"""
attached_architectures.
Class
-----------
R_Arch : learns regularization hyperparameter.
"""


import torch.nn as nn
from tools import norm_batch
import torch
Soft = nn.Softplus()
r=nn.ReLU()
class R_Arch(torch.nn.Module):

    def __init__(self,Arch):
        super(R_Arch, self).__init__()
        self.architecture=Arch
        if self.architecture =='lambda_Arch1_cvx':
            self.lamda_cvx=nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)
    def forward(self,H,x,xdeg):
        if self.architecture =='lambda_Arch1_cvx':
            lamda_cvx=r(self.lamda_cvx)
            return lamda_cvx,torch.zeros_like(lamda_cvx)
            
        
        
     
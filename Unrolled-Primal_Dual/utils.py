"""
utils.
Functions
-----------
activation primal : performs soft threshholding of a given vector.
activation dual : performs activation on dual branch.
SNR    : computes signal-to-noise ratio metric between a groundtruth and predicted signal.
TSNR   : computes truncated signal-to-noise ratio metric between a groundtruth and predicted signal.
"""
import torch
import numpy as np

def activation_primal(u, tau, factor=1.):
    zer=torch.zeros_like(u)
    return torch.sign(u) * torch.maximum(zer, torch.abs(u) - tau)


def activation_dual(u,y,sigma,rho):
    rho_vect=rho*torch.ones(u.size()[0]).type(torch.cuda.DoubleTensor)
    comp_vect=(torch.linalg.norm(u/sigma - y,dim=1) <= rho_vect).type(torch.cuda.DoubleTensor)
    
    u[comp_vect==True]=torch.zeros_like(u[comp_vect==True])
    u[comp_vect==False]=u[comp_vect==False]-sigma*(y+rho*torch.div(u[comp_vect==False]-sigma*y,torch.linalg.norm(u[comp_vect==False]-sigma*y,dim=1).unsqueeze(1).repeat(1,u[comp_vect==False].size()[1])))
    return u

def SNR(x_true,x_pred):
    n=0
    d=0
    sum_snr=0
    for i in range(x_true.size()[0]):
        n=torch.linalg.norm(x_true[i])
        d=torch.linalg.norm((x_true-x_pred)[i])
        sum_snr=sum_snr+20*torch.log10(n/d)
    return (sum_snr/x_true.size()[0])


def TSNR(x_true,x_pred):
    return 20*torch.log10(torch.linalg.norm(x_true[x_true!=0])/torch.linalg.norm(x_true[x_true!=0]-x_pred[x_true!=0]))

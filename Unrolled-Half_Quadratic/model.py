"""
Unfolded MM model classes.
Classes
-------
    Iter       : computes the estimate after each iteration
    Block      : one layer in unfolded architecture
    myModel    : Unfolded MM model
"""

import torch.nn as nn
import torch
from mode import MM, MMMG
from tools import phi_sfair,phi_scauchy,phi_swelsh,F,NormGradF,phi_sreweighting,phi_sgreen,phi_sl1,phi_sGMc,phi_sconvex,phi_sreweighting_cvx, SNR, norm_batch
from attached_architectures import R_Arch 
import matplotlib.pyplot as plt
import gc
import time
from tools import omega_sfair,omega_scauchy,omega_swelsh,omega_sgreen,omega_sGMc,omega_sreweighting,omega_sl1,omega_sreweighting_cvx
from torch.autograd import Variable
import cProfile
n = nn.Identity()
r=nn.ReLU()
soft = nn.Softplus()
class Iter(torch.nn.Module):
    """"
    Simulates one layer by performing one iteration and returning xk+1

    Attributes (to be thought about).......
    _________
    """

    def __init__(self, V1, V2, V3, n_in, n_out, Mk, dtype, mode,architecture_lambda):
      
        super(Iter, self).__init__()
        self.mode = mode
        if mode == "MM" :
            self.V1 = nn.Linear(n_in, n_out, bias=False)
            self.V1.weight = nn.Parameter(V1, requires_grad=False)
            self.V2 = nn.Linear(n_in, n_out, bias=False)
            self.V2.weight = nn.Parameter(V2, requires_grad=False)
            self.V3 = nn.Linear(3 * n_in, n_out, bias=False)
            self.V3.weight = nn.Parameter(V3, requires_grad=False)
            self.mat = nn.Linear(n_in, n_out, bias=False)
        if mode == "learning_lambda_MM" :
            self.architecture=R_Arch(architecture_lambda)
        
            
    def forward(self, x,x_precd, xdeg, Ht_x_degraded, xtrue ,delta_s_cvx,delta_s_ncvx, H, H_t, L, N, p, penalization_number_cvx,penalization_number_ncvx,lamda_cvx,lamda_ncvx):

        penal_name_cvx = 'phi_s' + str(penalization_number_cvx)
        penal_name_ncvx = 'phi_s' + str(penalization_number_ncvx)
        if self.mode == "MM" :
            first_branch = (n(self.V1(x) - Ht_x_degraded))
            second_branch = eval(penal_name_cvx + "(self.V2(x),delta_s_cvx)")
            second_branch1 = eval(penal_name_ncvx + "(self.V2(x),delta_s_ncvx)")
            concat = torch.cat((first_branch, second_branch,second_branch1), 1)
            glob = (self.V3(concat))
            glob = n(glob)
            glob = self.mat(glob)
            x = x - glob
        if self.mode == "learning_lambda_MM":
            lamda_cvx,lamda_ncvx=self.architecture(H,x,xdeg)
            first_branch= torch.transpose(torch.matmul(torch.mm(H_t,H),torch.transpose(x,0,1)),0,1) -Ht_x_degraded
            second_branch = eval(penal_name_cvx + "(x,delta_s_cvx)")
            second_branch1 = eval(penal_name_ncvx + "(x,delta_s_ncvx)")
            summ= lamda_cvx *second_branch + first_branch +lamda_ncvx*second_branch1
            inv_A = MM(x, xdeg, H,L,delta_s_cvx,delta_s_ncvx, lamda_cvx, lamda_ncvx,penalization_number_cvx,penalization_number_ncvx,self.mode)
            x = x -torch.bmm(inv_A,summ.unsqueeze(2)).squeeze()
        return x


class Block(torch.nn.Module):
    """
    Creates One layer in the network

    """

    def __init__(self, V1, V2, V3, n_in, n_out, Mk, dtype, mode,architecture_lambda):
        super(Block, self).__init__()
      
        self.Iter = Iter(V1, V2, V3, n_in, n_out, Mk, dtype, mode,architecture_lambda)

    def forward(self, x, xdeg, Ht_x_degraded,xtrue, delta_s,delta_s1, H, H_t, L, N, p, penalization_num,penalization_num1, xprecd,lamda_cvx,lamda_ncvx):
        """
        Computes the next iterate

        Parameters
        ----------
        x             (torch.nn.FloatTensor): previous iterate,
        Ht_x_degraded (torch.nn.FloatTensor): Ht*y
        """
        
        return self.Iter(x, xdeg, Ht_x_degraded,xtrue, delta_s,delta_s1, H, H_t, L, N, p, penalization_num,penalization_num1, xprecd,lamda_cvx,lamda_ncvx)

class myModel(torch.nn.Module):
    """"
    Creates the model.

    Attributes
    ----------
    Layers    (torch.nn.ModuleList object): list of the network's layers
    V1 (torch.FloatTensor): convolution filter corresponding to Ht*H
    b1 (torch.FloatTensor): bias term corresponding to -Ht*y
    V2 (torch.FloatTensor): convolution filter corresponding to L
    V3 (torch.FloatTensor): convolution filter corresponding to [Id|lambda*Lt]
    n_in    (float)       : size of input tensor to layers 1 and 2,Bkt | 2*n_in for 3
    n_out   (float)       : size of output tensor to layers 1, 2, 3 and Bk
    Mk      (float)       : Number of columns in Bk
    lamda   (float)       : Regularization term
    """

    def __init__(self, number_layers, V1, V2, V3, H, Ht, n_in, n_out, p, lamda_cvx,lamda_ncvx, delta_s_cvx,delta_s_ncvx, Mk, dtype, mode,
                 number_penalization_cvx,number_penalization_ncvx,architecture_lambda):
        super(myModel, self).__init__()

        self.Layers = nn.ModuleList()
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.H = H
        self.Ht = Ht
        self.n_in = n_in
        self.n_out = n_out
        self.Mk = Mk
        self.p = p
        self.lamda_cvx = lamda_cvx
        self.lamda_ncvx = lamda_ncvx
        self.delta_s_cvx = delta_s_cvx
        self.delta_s_ncvx = delta_s_ncvx
        self.dtype = dtype
        self.mode = mode
        self.number_penalization_cvx = number_penalization_cvx
        self.number_penalization_ncvx = number_penalization_ncvx
        self.architecture_lambda=architecture_lambda
        for i in range(number_layers):
            self.Layers.append(Block(V1, V2, V3, n_in, n_out, Mk, self.dtype, mode,architecture_lambda))
        
        
   
    def forward(self, x, xdeg, xtrue, Ht_x_degraded, mode,plot_F_gradF):
        """
        Computes the optimization problem solution given an initialization.

        Parameters
        ----------
        x             (torch.nn.FloatTensor): first iterate
        xdeg          (torch.nn.FloatTensor): y
        Ht_x_degraded (torch.nn.FloatTensor): Ht*y
        
        """
        for i, l in enumerate(self.Layers):
            if mode == "MM":
                inv_A = MM(x, xdeg,  self.H,self.V2, self.delta_s_cvx,self.delta_s_ncvx, self.lamda_cvx,self.lamda_ncvx,
                           self.number_penalization_cvx,self.number_penalization_ncvx,self.mode).type(self.dtype)
                inv_A=torch.squeeze(inv_A,0)
                self.Layers[i].Iter.mat.weight = torch.nn.Parameter(inv_A, requires_grad=False)
                x = self.Layers[i](x,x, xdeg, Ht_x_degraded,xtrue, self.delta_s_cvx,self.delta_s_ncvx, self.H, self.Ht, self.V2, self.n_in, self.p,self.number_penalization_cvx,self.number_penalization_ncvx, self.lamda_cvx,self.lamda_ncvx)

            if (mode == "learning_lambda_MM"):
                x = self.Layers[i](x,x, xdeg, Ht_x_degraded,xtrue, self.delta_s_cvx,self.delta_s_ncvx, self.H, self.Ht, self.V2, self.n_in, self.p,
                                   self.number_penalization_cvx,self.number_penalization_ncvx,self.lamda_cvx,self.lamda_ncvx)
        return x    
                   
            
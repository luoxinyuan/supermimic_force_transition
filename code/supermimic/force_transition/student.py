"""Causal student: only joint/IMU/action history and recurrent hidden state."""
import copy
import torch
from torch import nn
from ..networks import run_mlp

class ForceStudent(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.obs_norm=copy.deepcopy(teacher.obs_norm)
        self.decoder=copy.deepcopy(teacher.decoder).requires_grad_(False)
        self.encoder=nn.Sequential(nn.Linear(930,256),nn.ELU())
        self.history=nn.GRU(256,128,batch_first=True)
        self.head=nn.Linear(128,32)
    def sequence(self,obs,hidden=None):
        x=self.encoder(self.obs_norm(obs))
        x,hidden=self.history(x,hidden)
        return torch.tanh(self.head(x)),hidden
    def decode(self,obs,z):
        return run_mlp(self.decoder,torch.cat((z,self.obs_norm(obs)),-1))
    def forward(self,obs,hidden=None):
        z,hidden=self.sequence(obs[:,None],hidden)
        return self.decode(obs,z[:,0]),hidden

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ted_talk_data_feeder as ttdf

import numpy as np
from torch.nn.parameter import Parameter
from TED_data_location import ted_data_path, wordvec_path

class SyntacticSemanticEngine(nn.Module):
    '''
    Syntactic Semantic Engine is a new model for representing a dependency
    tree along with the distributed representations of the corresponding words.
    '''
    def __init__(self,dep_dict,pos_dict,GPUnum=0,sensedim=300,\
            w2vdim=300,nonlin=F.relu):
        '''
        To initiate, provide dictionaries that map to indices from dependency
        relations (dep_dict) and parts of speech (pos_dict).
        If GPUnum>=0, Cuda version of the tensors would be used. If you don't
        want to use GPU, set GPUdeviceNum=-1
        '''
        super(SyntacticSemanticEngine,self).__init__()
        # Size of the learnable parameters
        # Sense vector size
        self.s = sensedim
        # Word vector size
        self.N = w2vdim
        # Dependency vocabulary size
        self.d = len(dep_dict)
        # POS vocabulary size
        self.p = len(pos_dict)
        # Use GPU or not and device number
        self.gpu = GPUnum

        # Define the network parameters
        # Initiate dependency embedding for each dependency relation type
        self.D = {}
        for a_dep in dep_dict:
            if self.gpu<0:
                self.D[a_dep]=F.Linear(self.s,self.s)
            else:
                self.D[a_dep]=F.Linear(self.s,self.s).cuda(self.gpu)

        # Initiate POS Embedding for each POS type
        self.P = {}
        for a_pos in pos_dict:
            if self.gpu<0:
                self.P[a_pos]=F.Linear(self.N,self.s)
            else:
                self.P[a_pos]=F.Linear(self.N,self.s).cuda(self.gpu)
        
        # Set nonlinearity
        self.nonlin=nonlin
    
    def __process_node__(self,wvec,pos,dep,hin):
        '''
        Procedure to encode a single node in the tree
        '''
        # Input word vector
        if self.gpu < 0:
            xin = autograd.Variable(torch.Tensor(wvec))
        else:
            with torch.cuda.device(self.gpu):
                xin = autograd.Variable(torch.cuda.Tensor(wvec))
        u = self.nonlin(self.P[pos](xin)+hin)
        hout = self.nonlin(self.D[dep](u))
        return hout

    def encodetree(self,atree,glove_voc):
        for i,anode in enumerate(atree):
            if type(anode) == list:
                pass
            elif type(anode) == unicode or type(anode)==str:
                pass
                    

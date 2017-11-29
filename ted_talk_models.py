import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ted_talk_data_feeder as ttdf

import numpy as np
from torch.nn.parameter import Parameter
from TED_data_location import ted_data_path, wordvec_path

class multiLinear(nn.Module):
    '''
    An embedding module to learn "dic_len" number of affine transformations.
    Similar to linear but more than one pair of weight and biases.
    '''
    def __init__(self,dic_len,in_size,out_size):
        super(multiLinear,self).__init__()
        self.weight = Parameter(torch.randn(dic_len,out_size,in_size))
        self.bias = Parameter(torch.randn(dic_len,out_size))
    def forward(self,i,x):
        return torch.matmul(self.weight[i,:,:],x)+self.bias[i,:]

def __def_tensor__(gpunum,listobj):
    '''
    Helper Function.
    This function defines a tensor considering whether or not to use the GPU.
    '''
    if gpunum < 0:
        return autograd.Variable(torch.Tensor(listobj))
    else:
        with torch.cuda.device(gpunum):
            return autograd.Variable(torch.cuda.Tensor(listobj))

class SyntacticSemanticEngine(nn.Module):
    '''
    Syntactic Semantic Engine is a new model for representing a dependency
    tree along with the distributed representations of the corresponding words.
    '''
    def __init__(self,dep_dict,pos_dict,glove_voc,GPUnum=0,sensedim=300,\
        nonlin=F.relu):
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
        self.N = len(glove_voc['the'])
        # Dependency vocabulary size
        self.d = len(dep_dict)
        self.dep_dict = dep_dict
        # POS vocabulary size
        self.p = len(pos_dict)
        self.pos_dict = pos_dict
        # Save the word2vec dictionary
        self.glove_voc = glove_voc
        # Use GPU or not and device number
        self.gpu = GPUnum

        # Define the network parameters
        # Initiate dependency embedding for each dependency relation type
        if self.gpu < 0:
            self.D = multiLinear(self.d,self.s,self.s)
        else:
            self.D = multiLinear(self.d,self.s,self.s).cuda(self.gpu)
        # Initiate POS Embedding for each POS type
        if self.gpu < 0:
            self.P = multiLinear(self.p,self.N,self.s)
        else:
            self.P = multiLinear(self.p,self.N,self.s).cuda(self.gpu)
        
        # Set nonlinearity
        self.nonlin=nonlin
    
    def __process_node__(self,wvec,pos,dep,hin):
        '''
        Procedure to encode a single node in the tree
        '''
        # Input word vector
        xin = __def_tensor__(self.gpu,wvec)
        u = self.nonlin(self.P(pos,xin)+hin)
        hout = self.nonlin(self.D(dep,u))
        return hout

    def encodetree(self,atree):
        '''
        Recursively encodes a dependency tree to its embedding vector
        '''
        hout_sum = None
        for i,anode in enumerate(atree):
            if type(anode) == unicode or type(anode)==str:
                # lookahead if the next node is a subtree
                if i < len(atree)-1 and type(atree[i+1])==list:
                    # Next node is a subtree, process it first
                    hin = self.encodetree(atree[i+1])
                else:
                    # This node doesn't have any child. set hin to zero
                    hin = __def_tensor__(self.gpu,[0 for i in range(self.s)])
                # Compute the current node
                w,p,d = anode.strip().encode('ascii','ignore').split()
                hout = self.__process_node__(self.glove_voc[w],\
                    self.pos_dict[p],self.dep_dict[d],hin)
                # Add all the children values together
                if hout_sum is None:
                    hout_sum = hout
                else:
                    hout_sum+=hout
        return hout_sum

    def forward(self,minibatch):
        '''
        Produce the model output of a minibatch
        '''
        treevec = None
        for atree in minibatch:
            if not atree:
                raise IOError('Can not contain empty data')
            if not treevec:
                treevec = self.encodetree(atree)
            else:
                temp = self.encodetree(atree)
                treevec=torch.cat([treevec,temp],dim=0)
        return treevec

def __test_encodetree__():
    '''
    For testing purpose only. Checks the encodetree function in the SSE
    '''
    wdict = {'thank':[0.,0.,1.],'you':[0,0,0.5],'so':[0.1,0.5,0.7],\
    'much':[0,0.5,0.1],'chris':[0.1,0.2,0.1],',':[0.2,0.2,0.1],\
    '.':[0.4,0.1,0.3],'the':[0.5,0.5,0.5]}
    x = [u'thank VBP ROOT',[u'you PRP dobj',u'much RB advmod',\
    [u'so RB advmod'],u', , punct',u'chris FW dobj',u'. . punct']]
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    model = SyntacticSemanticEngine(dep_dict,pos_dict,wdict,\
        GPUnum=-1,sensedim=3)
    y = model.encodetree(x)

def test_with_multiLinear():
    '''
    For testing purpose only. This is a simple test to check if the
    multiLinear module works or not.
    Works perfectly.
    '''
    model = multiLinear(2,3,2)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    w1 = autograd.Variable(torch.Tensor([[1,0,0],[0,0,-1]]))
    b1 = autograd.Variable(torch.Tensor([1,1]))
    w2 = autograd.Variable(torch.Tensor([[0,-1,0],[1,0,0]]))
    b2 = autograd.Variable(torch.Tensor([2,-2]))
    for i in range(25000):
        x1 = autograd.Variable(torch.rand(3))
        y1 = torch.matmul(w1,x1)+b1
        x2 = autograd.Variable(torch.rand(3))
        y2 = torch.matmul(w2,x2)+b2
        model.zero_grad()
        loss = torch.sqrt(torch.pow(model(0,x1)-y1,2).sum()+\
            torch.pow(model(1,x2)-y2,2).sum())
        loss.backward()
        optimizer.step()
        print 'loss =',loss
    print model.state_dict()                    

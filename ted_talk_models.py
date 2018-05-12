import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence

import time
import numpy as np
from TED_data_location import ted_data_path, wordvec_path
import ted_talk_data_feeder as ttdf

class multiLinear(nn.Module):
    '''
    An embedding module to learn "dic_len" number of affine transformations.
    Similar to linear but more than one pair of weight and biases.
    '''
    def __init__(self,dic_len,in_size,out_size):
        super(multiLinear,self).__init__()
        self.weight = Parameter(torch.randn(dic_len,in_size,out_size))
        self.bias = Parameter(torch.randn(dic_len,1,out_size))
    def forward(self,i,x):
        return torch.matmul(x,self.weight[i,:,:])+self.bias[i]


def __def_tensor__(gpunum,listobj):
    '''
    Helper Function.
    This function defines a tensor considering whether or not to use the GPU.
    TODO: Set for deprecation. Utilize proper GPU handling technique mentioned
    in pytorch tutorial:
    http://pytorch.org/docs/0.3.1/notes/cuda.html#memory-management
    '''
    if gpunum < 0:
        return Variable(torch.Tensor([listobj]))
    else:
        with torch.cuda.device(gpunum):
            return Variable(torch.cuda.FloatTensor([listobj]))        

class LSTM_custom(nn.Module):
    '''
    A custom implementation of LSTM in pytorch. VERY slow
    '''
    def __init__(self,input_dim,hidden_dim):
        super(LSTM_custom,self).__init__()
        self.W_xi = nn.Linear(input_dim,hidden_dim)
        self.W_hi = nn.Linear(hidden_dim,hidden_dim)
        self.W_xf = nn.Linear(input_dim,hidden_dim)
        self.W_hf = nn.Linear(hidden_dim,hidden_dim)
        self.W_xg = nn.Linear(input_dim,hidden_dim)
        self.W_hg = nn.Linear(hidden_dim,hidden_dim)
        self.W_xo = nn.Linear(input_dim,hidden_dim)
        self.W_ho = nn.Linear(hidden_dim,hidden_dim)

    def forward(self,x,h,c):
        i = F.sigmoid(self.W_xi(x) + self.W_hi(h))
        f = F.sigmoid(self.W_xf(x) + self.W_hf(h))
        g = F.tanh(self.W_xg(x) + self.W_hg(h))
        o = F.sigmoid(self.W_xo(x)+self.W_ho(h))
        c_ = f*c + i*g
        h_ = o * F.tanh(c_)
        return h_,c_

class LSTM_TED_Rating_Predictor_Averaged(nn.Module):
    '''
    An LSTM based rating predictor. It expects to intake
    from ttdf.TED_Rating_Averaged_Dataset
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, gpuNum=-1):
        super(LSTM_TED_Rating_Predictor_Averaged, self).__init__()
        self.hidden_dim = hidden_dim
        #self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm = LSTM_custom(input_dim, hidden_dim)
        self.linear_rat1 = nn.Linear(hidden_dim, output_dim)
        self.linear_rat2 = nn.Linear(hidden_dim, output_dim)
        self.gpuNum = gpuNum
        self.hidden_0 = self.init_hidden() 


    def init_hidden(self):
        nullvec = np.zeros((1,self.hidden_dim)).astype(np.float32)
        return (ttdf.variablize(nullvec.copy(),self.gpuNum),
                ttdf.variablize(nullvec.copy(),self.gpuNum))

    def forward(self, minibatch):
        outrating = []
        for an_item in minibatch:
            # Feed through LSTM
            for i,an_input in enumerate(an_item['X']):                
                if i==0:
                    # set the first hidden
                    hx, cx = self.lstm(an_input,*self.hidden_0)
                else:
                    hx, cx = self.lstm(an_input, hx,cx)
            # Feed through Linear
            rat_layer1 = self.linear_rat1(hx).view(-1,1)
            rat_layer2 = self.linear_rat2(hx).view(-1,1)
            rat_layer = torch.cat((rat_layer1,rat_layer2),dim=1)            
            rat_scores = F.log_softmax(rat_layer, dim=1)
            outrating.append(rat_scores)
        return outrating

class SyntacticSemanticEngine(nn.Module):
    '''
    Syntactic Semantic Engine is a new model for representing a dependency
    tree along with the distributed representations of the corresponding words.
    '''
    def __init__(self,dep_dict,pos_dict,glove_voc,reduced=True,GPUnum=-1,
        sensedim=8,output_dim=14,activation=F.relu,final_activation=F.log_softmax):
        '''
        To initiate, provide dictionaries that map to indices from dependency
        relations (dep_dict) and parts of speech (pos_dict).
        If GPUnum>=0, Cuda version of the tensors would be used. If you don't
        want to use GPU, set GPUnum=-1
        If the output is not reduced, the final activation is applied over
        all the dependency trees in the input and then the result are stacked
        together and returned.
        If reduced, the output of each individual dependency tree
        is averaged and then the final activation function is
        applied
        '''
        super(SyntacticSemanticEngine,self).__init__()
        self.reduce = reduced
        # Size of the learnable parameters
        # Sense vector size
        self.s = sensedim
        # Output dimension
        self.outdim = output_dim
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
        self.D = multiLinear(self.d,self.s,self.s)
        self.P = multiLinear(self.p,self.N,self.s)
        self.linear = nn.Linear(self.s,self.outdim)
        # For GPU
        if self.gpu >= 0:
            with torch.cuda.device(self.gpu):
                self.D = self.D.cuda()
                self.P = self.P.cuda()
                self.linear = self.linear.cuda()
        
        # Set activations
        self.activation = activation
        self.final_activation = final_activation
    
    def __process_node__(self,w,p,d,hin):
        '''
        Procedure to encode a single node in the tree
        '''
        wpart_sum = np.array([0. for i in range(self.N)])
        wpart_count = 0
        # If the word is not available in the dictionary,
        # breaking into parts. If that doesn't work either,
        # just use zero.
        if w not in self.glove_voc and '-' in w:
            wparts = w.split('-')
        elif w not in self.glove_voc and '.' in w:
            wparts = w.split('.')
        else:
            wparts = [w]
        # acerage wordparts
        for wpart in wparts:
            if wpart in self.glove_voc:
                wpart_sum+=np.array(self.glove_voc[wpart])
                wpart_count+=1
        # Final wordvector
        wvec = wpart_sum/float(wpart_count) if wpart_count>0 else wpart_sum
        wvec = wvec.tolist()
        # Actual operation on a node
        xin = __def_tensor__(self.gpu,wvec)
        u = self.activation(self.P(self.pos_dict[p],xin)+hin)
        hout = self.activation(self.D(self.dep_dict[d],u))
        return hout

    def encodetree(self,atree):
        '''
        Recursively encodes a dependency tree to its embedding vector
        '''
        hout_sum = None
        count = 0.
        if not atree:
            raise IOError('Tree cannot be empty')
        # Loop over all children
        for i,anode in enumerate(atree):
            # If leaf node
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
                hout = self.__process_node__(w,p,d,hin)
                # Add all the children values together
                if hout_sum is None:
                    hout_sum = hout
                    count = 1.
                else:
                    hout_sum+=hout
                    count+=1.
        # Return the average of the children
        return torch.div(hout_sum,count)

    def __process_a_bag__(self,bag_of_dtree):
        bag_of_dtree_result = []
        if not self.reduce:
            # If not reduced, the final activation is applied over
            # all the dependency trees in the input and then the
            # results are stacked together and returned
            for atree in bag_of_dtree:
                if atree is None:
                    raise IOError('Can not contain empty data')
                # Calculate the embedding vector for each component
                # dimension of operation should be specified 
                # for log_softmax now
                if self.final_activation is F.log_softmax:
                    bag_of_dtree_result.append(self.final_activation(\
                        self.linear(self.encodetree(atree)),dim=1))
                else:
                    bag_of_dtree_result.append(self.final_activation(\
                        self.linear(self.encodetree(atree))))
            bag_of_dtree_result = torch.cat(bag_of_dtree_result,dim=0)
        else:
            # If reduced, the output of each individual dependency tree
            # is averaged and then the final activation function is
            # applied
            for atree in bag_of_dtree:
                if atree is None:
                    raise IOError('Can not contain empty data')
                # Calculate the embedding vector for each component
                bag_of_dtree_result.append(self.linear(self.encodetree(atree)))
            bag_of_dtree_result = torch.cat(bag_of_dtree_result,dim=0)
            # The final result is calculated as an average of the
            # bag of dependency trees
            if self.final_activation is F.log_softmax:
                bag_of_dtree_result = self.final_activation(\
                    bag_of_dtree_result.mean(dim=0),dim=0).view(1,-1)
            else:
                bag_of_dtree_result = self.final_activation(\
                    bag_of_dtree_result.mean(dim=0)).view(1,-1)
        return bag_of_dtree_result        

    def forward(self,bag_of_dtree):
        '''
        Produce the model output of a bag_of_dtree
        '''
        return self.__process_a_bag__(bag_of_dtree)

class RevisedTreeEncoder(nn.Module):
    '''
    A revised (as of Feb 26th, 2018) version of the Syntactic Semantic Engine.
    TODO: Test Thoroughly
    '''
    def __init__(self,dep_dict,pos_dict,glove_voc,reduced=True,GPUnum=-1,
        sensedim=8,output_dim=14,activation=F.relu,final_activation=F.log_softmax):
        '''
        To initiate, provide dictionaries that map to indices from dependency
        relations (dep_dict) and parts of speech (pos_dict). If the output is
        not reduced, the final activation is applied over all the dependency
        trees in the input and then the results are stacked together and 
        returned. If reduced, the output of each individual dependency tree is
        averaged and then the final activation function is applied
        '''
        super(RevisedTreeEncoder,self).__init__()
        self.reduce = reduced
        # Size of the learnable parameters
        self.s = sensedim
        self.outdim = output_dim
        # Word vector size
        self.N = len(glove_voc['the'])
        # Dependency vocabulary size
        self.d = len(dep_dict)
        # POS vocabulary size
        self.p = len(pos_dict)
        # Use GPU or not and device number
        self.gpu = GPUnum

        # Process and save the word2vec dictionary
        self.glove_tensor = torch.Tensor(glove_voc.values())
        self.glove_voc = {akey:i for i,akey in enumerate(glove_voc.keys())}

        # Zero wordvector for initialization
        self.wvec_init = torch.from_numpy(np.zeros((1,self.N),dtype=np.float32))
        self.hin_init = torch.from_numpy(np.zeros((1,self.s),dtype=np.float32))

        # Model parameters. W = Global word projector, D = dependency embedder
        # P = pos embedder and linear = output projector
        self.W = nn.Linear(self.N,self.s)
        # Transformation for dependency type
        self.D = nn.Embedding(self.d,self.s**2,max_norm=1.,norm_type=2.)
        self.c = nn.Embedding(self.d,self.s,max_norm=1.,norm_type=2.)
        # Transformation for POS
        self.P = nn.Embedding(self.p,self.s**2,max_norm=1.,norm_type=2.)
        self.b = nn.Embedding(self.p,self.s,max_norm=1.,norm_type=2.)
        # Output layer
        self.linear = nn.Linear(self.s,self.outdim)   

        # For GPU
        if self.gpu >= 0:
            # Move model parameters to GPU
            self.W = self.W.cuda(self.gpu)
            self.D = self.D.cuda(self.gpu)
            self.c = self.c.cuda(self.gpu)
            self.P = self.P.cuda(self.gpu)
            self.b = self.b.cuda(self.gpu)
            self.linear = self.linear.cuda(self.gpu)
            self.wvec_init = self.wvec_init.cuda(self.gpu)
            self.hin_init = self.hin_init.cuda(self.gpu)
            # Preallocate pos and dep dicts in a torch-preferred format
            self.dep_dict = {key:Variable(\
                torch.LongTensor([val])).cuda(self.gpu) for \
                key,val in dep_dict.items()}
            self.pos_dict = {key:Variable(\
                torch.LongTensor([val])).cuda(self.gpu) for \
                key,val in pos_dict.items()}
            self.glove_tensor = self.glove_tensor.cuda(self.gpu)
        else:
            # Preallocate pos and dep dicts in a torch-preferred format
            self.dep_dict = {key:Variable(\
                    torch.LongTensor([val])) for key,val in dep_dict.items()}
            self.pos_dict = {key:Variable(\
                    torch.LongTensor([val])) for key,val in pos_dict.items()}
        
        # Set activations
        self.activation = activation
        self.final_activation = final_activation
    
    def __build_wvec__(self,w):
        '''
        Construct the wordvectors in torch-preferred format
        '''
        wpart_sum = self.wvec_init
        wpart_count = 0
        # If the word is not available in the dictionary, try
        # breaking it into parts. If that doesn't work either,
        # just use zero.
        if w not in self.glove_voc and '-' in w:
            wparts = w.split('-')
        elif w not in self.glove_voc and '.' in w:
            wparts = w.split('.')
        else:
            wparts = [w]
        # average the wordparts
        for wpart in wparts:
            if wpart in self.glove_voc:
                voc_idx = self.glove_voc[wpart]
                if wpart_count == 0:
                    wpart_sum = self.glove_tensor[voc_idx,:].view(1,-1)
                else:
                    wpart_sum += self.glove_tensor[voc_idx,:].view(1,-1)
                wpart_count+=1
        # Final wordvector
        wvec = wpart_sum/float(wpart_count) if wpart_count>0 else wpart_sum
        return Variable(wvec)

    def __process_node__(self,w,p,d,hin):
        '''
        Procedure to encode a single node in the tree
        ''' 
        # Actual operation on a node
        xin = self.__build_wvec__(w)
        xproj = self.W(xin)
        # mapping for POS
        i_p = self.pos_dict[p]
        i_d = self.dep_dict[d]
        u = self.activation(torch.mm(xproj,self.P(i_p).view(self.s,self.s))\
            +self.b(i_p)+hin)
        hout = self.activation(torch.mm(u,self.D(i_d).view(self.s,self.s))+\
            self.c(i_d))
        return hout

    def encodetree(self,atree):
        '''
        Recursively encodes a dependency tree to its embedding vector
        '''
        hout_sum = None
        count = 0.
        if not atree:
            raise IOError('Tree cannot be empty')
        # Loop over all children
        for i,anode in enumerate(atree):
            # If leaf node
            if type(anode) == unicode or type(anode)==str:
                # lookahead if the next node is a subtree
                if i < len(atree)-1 and type(atree[i+1])==list:
                    # Next node is a subtree, process it first
                    hin = self.encodetree(atree[i+1])
                else:
                    # This node doesn't have any child. set hin to zero
                    hin = Variable(self.hin_init)
                # Compute the current node
                w,p,d = anode.strip().encode('ascii','ignore').split()
                hout = self.__process_node__(w,p,d,hin)
                # Add all the children values together
                if hout_sum is None:
                    hout_sum = hout
                    count = 1.
                else:
                    hout_sum+=hout
                    count+=1.
        # Return the average of the children
        return torch.div(hout_sum,count)

    def __process_a_bag__(self,bag_of_dtree):
        bag_of_dtree_result = []
        if not self.reduce:
            # If not reduced, the final activation is applied over
            # all the dependency trees in the input and then the
            # results are stacked together and returned
            for atree in bag_of_dtree:
                if atree is None:
                    raise IOError('Can not contain empty data')
                # Calculate the embedding vector for each component
                # dimension of operation should be specified 
                # for log_softmax now
                if self.final_activation is F.log_softmax:
                    bag_of_dtree_result.append(self.final_activation(\
                        self.linear(self.encodetree(atree)),dim=1))
                else:
                    bag_of_dtree_result.append(self.final_activation(\
                        self.linear(self.encodetree(atree))))
            bag_of_dtree_result = torch.cat(bag_of_dtree_result,dim=0)
        else:
            # If reduced, the output of each individual dependency tree
            # is averaged and then the final activation function is
            # applied
            for atree in bag_of_dtree:
                if atree is None:
                    raise IOError('Can not contain empty data')
                # Calculate the embedding vector for each component
                bag_of_dtree_result.append(self.linear(self.encodetree(atree)))
            bag_of_dtree_result = torch.cat(bag_of_dtree_result,dim=0)
            # The final result is calculated as an average of the
            # bag of dependency trees
            if self.final_activation is F.log_softmax:
                bag_of_dtree_result = self.final_activation(\
                    bag_of_dtree_result.mean(dim=0),dim=0).view(1,-1)
            else:
                bag_of_dtree_result = self.final_activation(\
                    bag_of_dtree_result.mean(dim=0)).view(1,-1)
        return bag_of_dtree_result        

    def forward(self,bag_of_dtree):
        '''
        Produce the model output of a bag_of_dtree
        '''
        return self.__process_a_bag__(bag_of_dtree)


# ----------------------------- Unit Test Codes -------------------------------

def __test_encodetree__():
    '''
    For testing purpose only. Checks the encodetree function in the SSE
    '''
    wdict = {'thank':[0.,0.,1.],'you':[0,0,0.5],'so':[0.1,0.5,0.7],\
    'much':[0,0.5,0.1],'chris':[0.1,0.2,0.1],',':[0.2,0.2,0.1],\
    '.':[0.4,0.1,0.3],'the':[0.5,0.5,0.5]}
    x = [u'thank VBP ROOT',[u'you PRP dobj',u'much RB advmod',\
    [u'so RB advmod'],u', , punct',u'chris FW dobj',u'. . punct']]
    x = [x,x]
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    model = SyntacticSemanticEngine(dep_dict,pos_dict,wdict,\
        GPUnum=-1,sensedim=3)
    print 'model input',x
    y = model(x)
    print 'model output',y

def __test_encodetree_revisedModel__():
    '''
    For testing purpose only. Checks the encodetree function in the
    revised tree encoder
    '''
    start_time = time.time()
    wdict = {'thank':[0.1,0.1,1.],'you':[0.2,1.,0.6],'so':[0.1,0.7,0.2],\
    'much':[0.9,0.1,0.1],'chris':[0.4,0.1,0.1],',':[0.6,0.2,0.6],\
    '.':[0.3,0.6,0.1],'the':[0.1,0.1,0.4]}
    dep_dict = {'ROOT':0,'dobj':1,'advmod':2,'punct':3}
    pos_dict = {'VBP':0,'PRP':1,'RB':2,',':3,'FW':4,'.':5}
    x = [u'thank VBP ROOT',[u'you PRP dobj',u'much RB advmod',\
    [u'so RB advmod'],u', , punct',u'chris FW dobj',u'. . punct']]
    x = [x,x]
    model = RevisedTreeEncoder(dep_dict,pos_dict,wdict,reduced=True,
        GPUnum=-1,sensedim=3,output_dim=2)
    optimizer = optim.Adam(model.parameters(),lr = 0.01)
    loss_fn = nn.KLDivLoss(size_average=False)
    gt = Variable(torch.Tensor([[0,1]]))
    # Training loop
    for iter in range(1000):
        model.zero_grad()
        log_probs = model(x)
        loss = loss_fn(log_probs,gt)
        loss.backward()
        optimizer.step()
        print 'loss:',loss.data[0]

    print 'model input',x
    y = model(x)
    print 'model output',torch.exp(y).data.numpy()
    print 'Evaluation time:',time.time() - start_time

def __test_encodetree_revisedModel_GPU__():
    '''
    For testing purpose only. Checks the encodetree function in the
    revised tree encoder
    '''
    start_time = time.time()
    wdict = {'thank':[0.1,0.1,1.],'you':[0.2,1.,0.6],'so':[0.1,0.7,0.2],\
    'much':[0.9,0.1,0.1],'chris':[0.4,0.1,0.1],',':[0.6,0.2,0.6],\
    '.':[0.3,0.6,0.1],'the':[0.1,0.1,0.4]}
    dep_dict = {'ROOT':0,'dobj':1,'advmod':2,'punct':3}
    pos_dict = {'VBP':0,'PRP':1,'RB':2,',':3,'FW':4,'.':5}
    x = [u'thank VBP ROOT',[u'you PRP dobj',u'much RB advmod',\
    [u'so RB advmod'],u', , punct',u'chris FW dobj',u'. . punct']]
    x = [x,x]
    model = RevisedTreeEncoder(dep_dict,pos_dict,wdict,reduced=True,
        GPUnum=0,sensedim=3,output_dim=2)
    optimizer = optim.Adam(model.parameters(),lr = 0.01)
    loss_fn = nn.KLDivLoss(size_average=False)
    gt = Variable(torch.Tensor([[0,1]])).cuda(0)
    # Training loop
    for iter in range(1000):
        model.zero_grad()
        log_probs = model(x)
        loss = loss_fn(log_probs,gt)
        loss.backward()
        optimizer.step()
        print 'loss:',loss.data[0]

    print 'model input',x
    y = model(x)
    print 'model output',torch.exp(y).cpu().data.numpy()    
    print 'Evaluation time:',time.time() - start_time



def __test_with_multiLinear__(gpunum=-1,nb_input = 3000,inp_dim = 300,
    out_dim = 20,nb_linear = 5,minibatch_size = 50,nb_iter = 25000,
    early_to_GPU=False,use_pinned=False):
    '''
    For testing purpose only. This is a simple test to check (1) if the
    multiLinear module works or not (2) Effect of GPU, and (3) Effect
    of pinned memory.
    If gpunum == -1 cpu will be used. Otherwise, GPU will be used
    Results:
    (1) Works perfectly.
    (2) Pinned memory has little to no effect or worse effect
    (3) Putting the variables early to GPU improves performance
    '''

    # Ideal weights (The learned weights should contain these values)
    w1 = torch.randn(inp_dim,out_dim)
    b1 = torch.randn(1,out_dim)
    w2 = torch.randn(inp_dim,out_dim)
    b2 = torch.randn(1,out_dim)

    # Inputs and outputs
    X = torch.randn(nb_input,inp_dim)
    y1 = torch.matmul(X,w1)+b1
    y2 = torch.matmul(X,w2)+b2

    # Pin inputs and outputs. Try enabling and disabling this
    if use_pinned:
        X = X.pin_memory()
        y1 = y1.pin_memory()
        y2 = y2.pin_memory()

    # Model: dictionary length x input size x output size
    model = multiLinear(nb_linear,inp_dim,out_dim)
    # Optimizer
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # Put model to gpu
    if gpunum >= 0:
        model = model.cuda(gpunum)
        if early_to_GPU:
            # putting the input output in GPU early
            X,y1,y2 = X.cuda(gpunum),y1.cuda(gpunum),y2.cuda(gpunum)

    # training steps
    for i in range(nb_iter):
        # Create minibatch
        batch_idx = np.random.rand(minibatch_size)*nb_input
        X_batch = X[batch_idx,:]
        y1_batch = y1[batch_idx,:]
        y2_batch = y2[batch_idx,:]

        # Put minibatch to GPU
        if gpunum >= 0 and not early_to_GPU:
            X_batch = X_batch.cuda(gpunum)
            y1_batch = y1_batch.cuda(gpunum)
            y2_batch = y2_batch.cuda(gpunum)

        # Put to Variable
        X_batch = Variable(X_batch)
        y1_batch = Variable(y1_batch)
        y2_batch = Variable(y2_batch)
        
        # Remove previous gradients
        model.zero_grad()

        # Pass through the model
        y1_out = model(0,X_batch)
        y2_out = model(1,X_batch)

        # Calculate the MSE loss
        loss = torch.sqrt(torch.pow(y1_out-y1_batch,2).sum()+\
            torch.pow(y2_out-y2_batch,2).sum())

        # Calculate gradients by backpropagation and update parameters
        loss.backward()
        optimizer.step()
        # print loss
        print 'iter:',i,'loss =',loss.data[0]
    print 'model:'
    print model.state_dict()
    print 'model targets:'
    print w1,b1,w2,b2

if __name__=='__main__':
    '''
    Test with multilinear
    '''
    start_time = time.time()

    # Time: 66.1393 sec
    # __test_with_multiLinear__(gpunum=-1,early_to_GPU=False,use_pinned=False)

    # Time: 68.41 sec
    # __test_with_multiLinear__(gpunum=-1,early_to_GPU=False,use_pinned=True)    
    
    # Time: 82.311 sec
    # __test_with_multiLinear__(gpunum=0,early_to_GPU=False,use_pinned=True)

    # Time: 82.281 sec
    # __test_with_multiLinear__(gpunum=0,early_to_GPU=False,use_pinned=False)

    # Time: 49.51 sec
    __test_with_multiLinear__(gpunum=0,early_to_GPU=True,use_pinned=False)

    # Time: 49.53 sec
    # __test_with_multiLinear__(gpunum=0,early_to_GPU=True,use_pinned=True)    

    print 'Elapsed Time:',time.time() - start_time

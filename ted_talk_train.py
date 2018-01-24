import os
import json
import itertools
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ted_talk_data_feeder as ttdf
import ted_talk_models as ttm
from TED_data_location import ted_data_path

def train_model(model,X,y,optimizer,loss_fn,log_filep):
    '''
    Trains the specified model with the provided dataset.
    '''
    pass

def train_sentencewise(output_folder = 'SSE_result/',
    sense_dim = 2,
    train_test_ratio = 0.75,
    loss_fn = nn.NLLLoss(),
    activation = F.log_softmax,
    model_outfile = 'model_weights.pickle',
    loss_outfile = 'lossfile.txt',
    useGPU = False,
    gpunum = -1):
    '''
    Procedure to train the SSE with sentences followed by laughter or 
    non-laughter tags.
    '''    

    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    np.random.shuffle(train_id)
    np.random.shuffle(test_id)

    # Reading Vocabs
    print 'Reading Vocabs'
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    w2v = ttdf.read_crop_glove()
    # Initialize the model
    model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,w2v,\
        GPUnum=gpunum,sensedim=sense_dim,final_activation=activation)
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    print 'Model Loaded'
    
    # Save the parameters of the function call. It allows me to audit the models
    with open(os.path.join(outpath,'params.txt'),'wb') as fparam:
        fparam.write('sense_dim={}'.format(sense_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('loss_fn={}'.format(loss_fn.__repr__())+'\n')
        fparam.write('activation={}'.format(activation.__repr__())+'\n')
        fparam.write('model_outfile={}'.format(model_outfile)+'\n')
        fparam.write('loss_outfile={}'.format(loss_outfile)+'\n')
        fparam.write('gpunum={}'.format(gpunum)+'\n')
        fparam.write('Optimizer_name={}'.format(optimizer.__repr__())+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')

    iter = 0
    # Write the loss in file
    with open(os.path.join(outpath,loss_outfile),'wb') as f:
        for atalk in train_id:
            print 'training',atalk
            # Feed the whole talk as a minibatch
            minibatch = [(adep,label) for adep,label,_,_ in \
                ttdf.generate_dep_tag(atalk)]
            X,y = zip(*minibatch)
            label = ttm.def_tensor(gpunum,y)
            # Clear gradients from previous iterations
            model.zero_grad()
            # Forward pass through the model
            model_out = model(X)
            import pdb;pdb.set_trace()
            # Calculate the loss
            loss = loss_fn(model_out,y)
            # Backpropagation of the gradients
            loss.backward()
            # Parameter update
            optimizer.step()
            print 'Loss:',loss.data[0]
    # Save the model
    model_filename = os.path.join(outpath,model_outfile)
    torch.save(model.cpu(),open(model_filename,'wb'))



if __name__=='__main__':
    train_sentencewise()


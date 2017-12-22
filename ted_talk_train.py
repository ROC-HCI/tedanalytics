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


def train_sentencewise(output_folder = 'SSE_result/',
    sense_dim = 2,
    train_test_ratio = 0.75,
    loss_fn = 'nllloss',
    activation = 'logsoftmax',
    model_outfile = 'model_weights.pickle',
    loss_outfile = 'lossfile.txt',
    useGPU = False,
    devicenumber = -1):
    '''
    Procedure to train the SSE with sentences followed by laughter or 
    non-laughter tags.
    '''    
    outpath = os.path.join(ted_data_path,output_folder)
    # If the output folder is not ready, create it
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    np.random.shuffle(train_id)
    np.random.shuffle(test_id)

    # Reading Vocabs
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    w2v = ttdf.read_crop_glove()
    # Initialize the model
    model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,w2v,\
        GPUnum=devicenumber)
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    
    # Save the parameters of the function call. It allows me to audit the models
    with open(os.path.join(outpath,'params.txt'),'wb') as fparam:
        fparam.write('sense_dim={}'.format(sense_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('loss_fn={}'.format(loss_fn)+'\n')
        fparam.write('activation={}'.format(activation)+'\n')
        fparam.write('model_outfile={}'.format(model_outfile)+'\n')
        fparam.write('loss_outfile={}'.format(loss_outfile)+'\n')
        fparam.write('devicenumber={}'.format(devicenumber)+'\n')
        fparam.write('Optimizer_name={}'.format(optimizer.__repr__())+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')

    iter = 0
    # Write the loss in file
    with open(os.path.join(outpath,loss_outfile),'wb') as f:
        for atalk in train_id:
            # Feed the whole talk as a minibatch
            minibatch = [(adep,label) for adep,label,_,_ in \
                ttdf.generate_dep_tag(atalk)]
            X,y = zip(*minibatch)
            model_out = model(X)


if __name__=='__main__':
    train_sentencewise()


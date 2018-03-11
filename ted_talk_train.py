import os
import time
import json
import glob
import numpy as np
import cPickle as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ted_talk_data_feeder as ttdf
import ted_talk_models as ttm
from TED_data_location import ted_data_path

def __build_SSE__(reduced_val,sense_dim=14,gpunum=-1,\
    final_activation=F.log_softmax):
    '''
    Helper function to initiate a Syntactic-Semantic-Engine and 
    initiates with data.
    If reduced, the output of each individual dependency tree
    is averaged and then the final activation function is
    applied. 
    '''
    # Reading Vocabs
    print 'Reading Vocabs'
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    wvec = ttdf.read_crop_glove()
    # Initialize the model
    model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,wvec,\
        reduced=reduced_val,GPUnum=gpunum,sensedim=sense_dim,\
        final_activation=final_activation)
    print 'Model Initialized'
    return model

def __build_RTE__(reduced_val,sense_dim=2,output_dim=14,gpunum=-1,\
    final_activation=F.log_softmax):
    '''
    Helper function to initiate a Revised Syntactic-Semantic-Engine and
    initiates with data.
    If reduced, the output of each individual dependency tree
    is averaged and then the final activation function is
    applied. 
    '''
    # Reading Vocabs
    print 'Reading Vocabs'
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    wvec = ttdf.read_crop_glove()
    # Initialize the model
    model = ttm.RevisedTreeEncoder(dep_dict,pos_dict,wvec,\
        reduced=reduced_val,GPUnum=gpunum,sensedim=sense_dim,\
        output_dim=output_dim,final_activation=final_activation)
    print 'Model Initialized'
    return model

def train_model(model, feeder,
    output_folder = 'SSE_result/',
    train_test_ratio = 0.85,
    loss_fn_name = nn.KLDivLoss,
    optim_fn_name = optim.Adam,
    learning_rate = 0.01,
    model_outfile = 'model_weights.pkl',
    output_log = 'train_logfile.txt',
    max_data = np.inf,
    max_iter = 3):
    '''
    Given an initialized (but not trained) model and a feeder function
    (to supply the data in a format appropriate for the model and to 
    supply the ground truth in a format appropriate for the loss and model
    output), this function will train the model. The feeder function takes
    a talk_id and (optionally) a gpu number for cases when the model
    is put into a GPU.

    By default, KL Divergence loss is used.
    This function outputs a logfile, the trained model file and a misc
    file containing the loss function and the optimizer function.
    '''
    outpath = os.path.join(ted_data_path,output_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    np.random.shuffle(train_id)
    np.random.shuffle(test_id)

    # Use sum, not average.
    loss_fn = loss_fn_name(size_average=False)
    # Initialize the optimizer
    optimizer = optim_fn_name(model.parameters(),lr = learning_rate)

    # Save the parameters of the function call. It allows me to audit the models
    with open(os.path.join(outpath,output_log),'wb') as fparam:
        fparam.write('sense_dim={}'.format(model.s)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('activation={}'.format(model.activation.__repr__())+'\n')
        fparam.write('final_activation={}'.format(\
            model.final_activation.__repr__())+'\n')
        fparam.write('learning_rate={}'.format(learning_rate)+'\n')
        fparam.write('model_outfile={}'.format(model_outfile)+'\n')
        fparam.write('gpunum={}'.format(model.gpu)+'\n')
        fparam.write('Optimizer_name={}'.format(optimizer.__repr__())+'\n')
        fparam.write('Loss_name={}'.format(loss_fn.__repr__())+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')
        losslist = []
        # Iteration
        for iter in range(max_iter):
            # Shuffle the training batch
            np.random.shuffle(train_id)
            # Loop over one datapoint at a time
            for i,atalk in enumerate(train_id):
                if i > max_data:
                    break

                # Get the input and the ground truth
                all_deptree,rating_t = feeder(atalk,model.gpu)

                # Clear gradients from previous iterations
                model.zero_grad()
                # Forward pass through the model
                log_probs = model(all_deptree)
                # Calculate the loss
                loss = loss_fn(log_probs,rating_t)

                # Backpropagation of the gradients
                loss.backward()
                # Parameter update
                optimizer.step()

                # Logging the current status
                lossval = loss.data[0]

                # Save the loss in the last iteration
                # This is to compute the average training loss and the
                # model performance over the training data.
                if iter == max_iter - 1:
                    losslist.append(lossval)
                # Show status
                status_msg =  'training:'+str(atalk)+', Loss:'+\
                    str(lossval)+', iteration:'+str(iter)
                print status_msg
                fparam.write(status_msg + '\n')
        # Write the average loss of last iteration
        status_msg = 'Average Loss in last iteration:{}\n'.format(np.mean(losslist))
        print status_msg
        fparam.write(status_msg)
    # Save the model
    model_filename = os.path.join(outpath,model_outfile)
    torch.save(model.cpu(),open(model_filename,'wb'))

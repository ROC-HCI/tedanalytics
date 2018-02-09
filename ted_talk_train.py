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

def train_humor_sentencewise(output_folder = 'SSE_result/',
    sense_dim = 2,
    train_test_ratio = 0.75,
    final_activation = F.log_softmax,
    model_outfile = 'model_weights.pickle',
    output_log = 'logfile.txt',
    gpunum = -1,
    max_iter=np.inf):
    '''
    Procedure to train the SSE with sentences followed by laughter or 
    non-laughter tags.
    '''    
    outpath = os.path.join(ted_data_path,output_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    np.random.shuffle(train_id)
    np.random.shuffle(test_id)

    # Reading Vocabs
    print 'Reading Vocabs'
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    wvec = ttdf.read_crop_glove()
    # Initialize the model
    model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,wvec,\
        GPUnum=gpunum,sensedim=sense_dim,final_activation=final_activation)
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(),lr = 0.01)
    # Use Negative Log Likelihood Loss
    loss_fn = nn.NLLLoss(size_average=False)
    # Show status
    print 'Model Loaded'
    
    # Save the parameters of the function call. It allows me to audit the models
    with open(os.path.join(outpath,output_log),'wb') as fparam:
        fparam.write('sense_dim={}'.format(sense_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('activation={}'.format(model.activation.__repr__())+'\n')
        fparam.write('final_activation={}'.format(\
            model.final_activation.__repr__())+'\n')
        fparam.write('model_outfile={}'.format(model_outfile)+'\n')
        fparam.write('gpunum={}'.format(gpunum)+'\n')
        fparam.write('Optimizer_name={}'.format(optimizer.__repr__())+'\n')
        fparam.write('Loss_name={}'.format(loss_fn.__repr__())+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')
        iter = 0
        # Write the loss in file
        for i,atalk in enumerate(train_id):
            if i > max_iter:
                break
            # Feed the whole talk as a minibatch
            minibatch = [(adep,label) for adep,label,_,_ in \
                ttdf.generate_dep_tag(atalk)]
            X,y = zip(*minibatch)
            # Construct the label tensor
            if gpunum < 0:
                labels_t = autograd.Variable(torch.LongTensor(y))
            else:
                with torch.cuda.device(gpunum):
                    labels_t = autograd.Variable(torch.cuda.LongTensor(y))
            
            # Clear gradients from previous iterations
            model.zero_grad()
            # Forward pass through the model
            log_probs = model(X)
            # Calculate the loss
            loss = loss_fn(log_probs,labels_t)
            # Backpropagation of the gradients
            loss.backward()
            # Parameter update
            optimizer.step()

            # Show status
            lossval = loss.data[0]
            status_msg =  'training:'+str(atalk)+', Loss:'+str(lossval)+\
                ' #of_trees:'+str(len(y))+' avg_loss_per_tree:'+\
                str(lossval/len(y))
            print status_msg
            fparam.write(status_msg + '\n')
    # Save the model
    model_filename = os.path.join(outpath,model_outfile)
    torch.save(model.cpu(),open(model_filename,'wb'))

def train_ratings(output_folder = 'SSE_result/',
    sense_dim = 14,
    train_test_ratio = 0.75,
    final_activation = F.log_softmax,
    model_outfile = 'model_weights.pickle',
    output_log = 'logfile.txt',
    gpunum = -1,
    max_iter=np.inf):
    '''
    Procedure to train the SSE with ratings. KL Divergence loss is used
    '''
    outpath = os.path.join(ted_data_path,output_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    np.random.shuffle(train_id)
    np.random.shuffle(test_id)

    # Reading Vocabs
    print 'Reading Vocabs'
    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
    wvec = ttdf.read_crop_glove()

    # Initialize the model
    model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,wvec,\
        GPUnum=gpunum,sensedim=sense_dim,final_activation=final_activation)
    # Use Negative Log Likelihood Loss
    loss_fn = nn.KLDivLoss(size_average=False)    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(),lr = 0.01)
    print 'Model Loaded'

    # Save the parameters of the function call. It allows me to audit the models
    with open(os.path.join(outpath,output_log),'wb') as fparam:
        fparam.write('sense_dim={}'.format(sense_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('activation={}'.format(model.activation.__repr__())+'\n')
        fparam.write('final_activation={}'.format(\
            model.final_activation.__repr__())+'\n')
        fparam.write('model_outfile={}'.format(model_outfile)+'\n')
        fparam.write('gpunum={}'.format(gpunum)+'\n')
        fparam.write('Optimizer_name={}'.format(optimizer.__repr__())+'\n')
        fparam.write('Loss_name={}'.format(loss_fn.__repr__())+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')
        iter = 0
        # Write the loss in file
        for i,atalk in enumerate(train_id):
            if i > max_iter:
                break
            print 'training',atalk
            # All the dependency trees and the rating
            all_deptree,rating = ttdf.get_dep_rating(atalk)
            y = [vals for keys,vals in sorted(rating.items())]
            # Construct ground truth tensor
            if gpunum < 0:
                rating_t = autograd.Variable(torch.FloatTensor(y))
            else:
                with torch.cuda.device(gpunum):
                    rating_t = autograd.Variable(torch.cuda.FloatTensor(y))
            # Clear gradients from previous iterations
            model.zero_grad()
            
            # Forward pass through the model
            log_probs = model(all_deptree,reduce=True)
            # Calculate the loss
            loss = loss_fn(log_probs,rating_t)

            # Backpropagation of the gradients
            loss.backward()
            # Parameter update
            optimizer.step()
            
            # Show status
            lossval = loss.data[0]
            status_msg =  'training:'+str(atalk)+', Loss:'+str(lossval)
            print status_msg
            fparam.write(status_msg + '\n')
    # Save the model
    model_filename = os.path.join(outpath,model_outfile)
    torch.save(model.cpu(),open(model_filename,'wb'))


if __name__=='__main__':
    import time
    start_time = time.time()
    np.random.seed(0)
    #train_humor_sentencewise(gpunum=-1,max_iter=10)
    train_ratings(gpunum=-1,max_iter=10)
    print time.time() - start_time



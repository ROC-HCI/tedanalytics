import os
import time
import json
import glob
import numpy as np
import cPickle as cp

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

def train_recurrent_models(
    dataset_type = 'word-only',
    firstThresh = 50.,
    secondThresh = 50.,
    scale_rating = True,
    minibatch_size = 5,
    hidden_dim = 128,
    modality=['word','audio','pose','face'],
    output_folder = 'TED_stats/',
    train_test_ratio = 0.85,
    learning_rate = 0.01,
    max_iter_over_dataset = 20,
    GPUnum = -1):
    '''
    Trains a GRU or LSTM model using TED_Rating_Averaged_Dataset
    '''
    old_time = time.time()
    # Prepare the output folder and its contents
    outpath = os.path.join(ted_data_path,output_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    output_log = 'LSTM_log_{0}_{1}_{2}_{3}_{4}_{5}_{6}.txt'.format(dataset_type,\
        scale_rating,''.join([m[0] for m in modality]),\
        firstThresh,secondThresh,hidden_dim,max_iter_over_dataset)
    outlogfile = os.path.join(outpath,output_log)
    model_filename = os.path.join(outpath,output_log.replace('LSTM_log',
        'LSTM_').replace('.txt','.model'))

    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    if dataset_type == 'averaged':
        train_dataset = ttdf.TED_Rating_Averaged_Dataset(data_indices=train_id,
            firstThresh = firstThresh, secondThresh = secondThresh,
            scale_rating = scale_rating,modality=modality)
    elif dataset_type == 'streamed':
        train_dataset = ttdf.TED_Rating_Streamed_Dataset(data_indices=train_id,
            firstThresh = firstThresh, secondThresh = secondThresh,
            scale_rating = scale_rating,modality=modality)
    elif dataset_type == 'word-only':
        train_dataset = ttdf.TED_Rating_wordonly_indices_Dataset(
            data_indices=train_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating)

    # Constructing minibaches. With lots of experiment, finally using
    # pytorch's dataloader approach as that appears to be the fastest
    # way for iterating over the data
    ##################### DEBUG 1 ###############
    # oldtime = time.time()
    #############################################
    # Option 1
    #minibatch_iter = ttdf.get_minibatch_iter(train_dataset,\
    #    minibatch_size,GPUnum)
    # Option 2
    #minibatch_iter = ttdf.get_minibatch_iter_pooled(train_dataset,\
    #    minibatch_size,GPUnum)
    # Option 3:
    minibatch_iter = ttdf.get_data_iter_simple(train_dataset,\
        batch_size=minibatch_size,gpuNum=GPUnum)
    ##################### DEBUG 2 ###############
    # print 'Elapsed time:',time.time()-oldtime
    #############################################
    for x in minibatch_iter:
        print type(x)
    import pdb; pdb.set_trace()  # breakpoint 1ba0884e //

    train_datalen = len(train_dataset)
    print 'Dataset Length:',train_datalen

    # Build the model
    model = ttm.LSTM_TED_Rating_Predictor_Averaged(
        train_dataset.dims,hidden_dim,len(train_dataset.ylabel),GPUnum)
    ttdf.gputize(model,GPUnum)

    # lossfunction
    loss_fn = nn.NLLLoss()
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    # Training time
    model.train()

    # Training loop
    data_count = 0
    losslist=[]
    # Save the parameters of the function call. It allows me to audit the models
    with open(outlogfile,'wb') as fparam:
        fparam.write('dataset_type={}'.format(dataset_type)+'\n')
        fparam.write('scale_rating={}'.format(scale_rating)+'\n')
        fparam.write('firstThresh={}'.format(firstThresh)+'\n')
        fparam.write('secondThresh={}'.format(secondThresh)+'\n')
        fparam.write('sense_dim={}'.format(hidden_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('learning_rate={}'.format(learning_rate)+'\n')
        fparam.write('model_outfile={}'.format(model_filename)+'\n')
        fparam.write('gpunum={}'.format(GPUnum)+'\n')
        fparam.write('modality={}'.format(json.dumps(modality))+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')
        while True:
            # form minibatch
            minibatch = minibatch_iter.next()

            # Multiple run on a minibatch
            # for i_ in range(5):
            # Clear gradients from previous iterations
            model.zero_grad()

            # Forward pass through the model
            log_probs = model(minibatch)

            # Calculate the loss
            loss = __compute_loss__(log_probs,minibatch,loss_fn)

            # Backpropagation of the gradients
            lossval = loss.clone()
            loss.backward()
            # Parameter update
            optimizer.step()

            # Logging the current status
            data_count += len(minibatch)
            ratio_trained = float(data_count)/float(train_datalen)
            if ratio_trained > max_iter_over_dataset:
                break
            lossval=lossval.cpu()
            lossval = lossval.data[0]
            if ratio_trained > 1:
                losslist.append(lossval)
            status_msg =  'training: ,'+' Loss:'+\
                str(lossval)+', iteration percent:'+str(ratio_trained*100)
            print status_msg
            fparam.write(status_msg + '\n')
        
        # Write the average loss of last iteration
        status_msg = 'Average Loss after first iteration through dataset:{}\n'\
            .format(np.nanmean(losslist))
        print status_msg
        fparam.write(status_msg)
        fparam.write('Time:'+str(time.time() - old_time)+'\n')
    print 'Total time:',time.time() - old_time

    # Save the model
    torch.save(model.cpu(),open(model_filename,'wb'))


def __compute_loss__(log_probs,minibatch,loss_fn):
    loss = None
    for i,an_item in enumerate(minibatch):
        label = an_item['Y'].view(-1)
        if i==0:
            loss = loss_fn(log_probs[i],label)
        else:
            loss = torch.cat((loss,loss_fn(log_probs[i],label)),dim=0)    
    loss = torch.mean(loss)
    return loss


if __name__=='__main__':
    train_recurrent_models()

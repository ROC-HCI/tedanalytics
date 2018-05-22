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
    minibatch_size = 100,
    hidden_dim = 256,
    modality=['word','audio','pose','face'],
    output_folder = 'TED_stats/',
    train_test_ratio = 0.90,
    learning_rate = 0.01,
    max_iter_over_dataset = 1000,
    GPUnum = 0):
    '''
    Trains a GRU or LSTM model using TED_Rating_Averaged_Dataset
    Name of all modalities: modality=['word','audio','pose','face'],
    '''
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
            secondThresh = secondThresh,scale_rating = scale_rating,
            flatten_sentence=False)
    train_datalen = len(train_dataset)
    print 'Dataset Length:',train_datalen

    # Prepare the output folder and its contents
    outpath = os.path.join(ted_data_path,output_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    output_log = 'LSTM_log_ty{0}_s{1}_{2}_{3}_{4}_hid{5}_it{6}_tr{7}_bat{8}.txt'.format(\
        dataset_type,scale_rating,''.join([m[0] for m in modality]),\
        firstThresh,secondThresh,hidden_dim,max_iter_over_dataset,\
        train_datalen,minibatch_size)
    outlogfile = os.path.join(outpath,output_log)
    model_filename = os.path.join(outpath,output_log.replace('LSTM_log',
        'LSTM_').replace('.txt','.model'))

    # Preparing file to save the parameters and status
    with open(outlogfile,'wb') as fparam:    
        fparam.write('dataset_type={}'.format(dataset_type)+'\n')
        fparam.write('dataset_len={}'.format(train_datalen)+'\n')
        fparam.write('minibat_len={}'.format(minibatch_size)+'\n')
        fparam.write('scale_rating={}'.format(scale_rating)+'\n')
        fparam.write('firstThresh={}'.format(firstThresh)+'\n')
        fparam.write('secondThresh={}'.format(secondThresh)+'\n')
        fparam.write('hidden_dim={}'.format(hidden_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('learning_rate={}'.format(learning_rate)+'\n')
        fparam.write('model_outfile={}'.format(model_filename)+'\n')
        fparam.write('gpunum={}'.format(GPUnum)+'\n')
        fparam.write('modality={}'.format(json.dumps(modality))+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')

        # Build the model
        print 'Loading Model (gputizing wvec) ...'
        model = ttm.LSTM_TED_Rating_Predictor_wordonly(
            hidden_dim=hidden_dim,output_dim=len(train_dataset.ylabel),
            wvec_vals=train_dataset.wvec_map.w2v_vals,gpuNum=GPUnum)
        ttdf.gputize(model,GPUnum)
        # Training time
        model.train()        
        print 'done'
        # lossfunction
        loss_fn = nn.BCEWithLogitsLoss()
        # Initialize the optimizer
        optimizer = optim.Adagrad(model.parameters(),lr = learning_rate)        
 
        # Iteration over the dataset
        data_count = 0
        old_time = time.time()
        for an_iter in range(max_iter_over_dataset):
            # Constructing minibaches. With lots of experiment, finally using
            # the first option as it is fast and convenient
            ##################### DEBUG 1 ###############
            # oldtime = time.time()
            #############################################
            # Option 1
            minibatch_iter = ttdf.get_minibatch_iter(train_dataset,\
                minibatch_size,GPUnum)
            # Option 2
            #minibatch_iter = ttdf.get_minibatch_iter_pooled(train_dataset,\
            #    minibatch_size,GPUnum)
            # Option 3: Note: The data is packed
            # minibatch_iter = ttdf.get_data_iter_simple(train_dataset,\
            #    batch_size=minibatch_size,gpuNum=GPUnum)
            ##################### DEBUG 2 ###############
            # print 'Elapsed time:',time.time()-oldtime
            #############################################

            # Training Loop
            losslist=[]
            for i,minibatch in enumerate(minibatch_iter):
                model.zero_grad()
                # Forward pass through the model                
                log_probs = model(minibatch)

                # Calculate the loss
                loss = __compute_loss__(log_probs,minibatch,loss_fn)
                # Backpropagation of the gradients
                lossval = loss.cpu().data.numpy()
                loss.backward()
                # Parameter update
                optimizer.step()

                # Logging the current status
                data_count += len(minibatch)
                ratio_trained = float(data_count)/float(train_datalen)
                losslist.append(lossval)
                status_msg='train:{0},Loss:{1:0.6},batch:{2}'
                status_msg+='Data_per:{3:0.2}%,Data_count:{4},iter_time:{5:0.4}'
                status_msg=status_msg.format(an_iter,lossval,i,\
                    ratio_trained*100,data_count,time.time()-old_time)

                print status_msg
                fparam.write(status_msg + '\n')
            # Write the average loss of last iteration
            meanloss = np.nanmean(losslist)
            status_msg = 'Average loss:{0}\n'.format(meanloss)
            print status_msg
            fparam.write(status_msg)
            fparam.flush()

            # # Save the model every iteration for safety
            # if an_iter%10==9:
            #     torch.save(model.cpu(),open(model_filename,'wb'))
            #     ttdf.gputize(model,GPUnum)

        # Save the model
        torch.save(model.cpu(),open(model_filename,'wb'))

def __compute_loss__(log_probs,minibatch,loss_fn):
    losslist = []
    count=0.    
    for i,an_item in enumerate(minibatch):
        losslist.append(loss_fn(log_probs[i],an_item['Y']))
        count+=1.
    loss = reduce(torch.add,losslist)/count

    return loss


if __name__=='__main__':
    train_recurrent_models()

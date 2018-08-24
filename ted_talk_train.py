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
    loss_fn = model.loss_fn
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
    model.save(open(model_filename,'wb'))

def train_recurrent_models(
    dataset_type = 'depposwordprosody',
    firstThresh = 50.,
    secondThresh = 50.,
    scale_rating = True,
    flatten_sentence = False,
    minibatch_size = 10,
    hidden_dim = 128,
    output_folder = 'TED_models/',
    train_test_ratio = 0.90,
    optimizer_fn = optim.Adam,
    learning_rate = 0.001,
    weight_decay = 0.00033,
    dropconnect = 0.2,
    max_iter_over_dataset = 48,
    GPUnum = 0):
    '''
    Trains the LSTM models using sequential datasets.
    
    **Currently only 'word-only', 'deppos', 'depposword'
    and 'depposwordprosody' dataset is allowed.**
    
    Output:
    1) A log file LSTM_log_ ... <random_number>.txt
    2) A model weight file LSTM_model_ ... <random_number>.model
    3) A model weight file LSTM_model_ ... <random_number>_final.model
    
    The random_number part for these two corresponding files will be same
    but they will be different in different run. The first file contains
    the training/testing status. The second file contains the neural weights
    of the LSTM network. The third file contains the final model weights after
    all iterations are done.

    Note: The other model file (file#2) is saved only when the test loss
    becomes lower than any previous test loss value.

    Note 2: the model files only store the state_dict of the network. 
    So, in the loading time, the model must be initiated correctly from the
    code BEFORE loading the weights from disk. The class name from which
    the model is instantiated is saved in the LSTM_log as "modelclassname".
    '''
    # Prepare trining and test split
    train_id,test_id = ttdf.split_train_test(train_test_ratio)
    # Select correct dataset
    if dataset_type == 'word-only':
        ################ DEBUG * REMOVE ###############
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # train_id = train_id[:10]
        # test_id = test_id[:2]
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        ###############################################
        train_dataset = ttdf.TED_Rating_wordonly_indices_Dataset(
            data_indices=train_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            flatten_sentence=flatten_sentence,gpuNum=GPUnum)
        test_dataset = ttdf.TED_Rating_wordonly_indices_Dataset(
            data_indices=test_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            flatten_sentence=flatten_sentence,gpuNum=GPUnum) 
    elif dataset_type == 'deppos':
        train_dataset = ttdf.TED_Rating_depPOSonly_indices_Dataset(
            data_indices=train_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            gpuNum=GPUnum)
        test_dataset = ttdf.TED_Rating_depPOSonly_indices_Dataset(
            data_indices=test_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            gpuNum=GPUnum)
    elif dataset_type == 'depposword':
        wvec=ttdf.wvec_index_maker(gpuNum=GPUnum)
        train_dataset = ttdf.TED_Rating_depPOSonly_indices_Dataset(
            data_indices=train_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            wvec_index_maker=wvec,gpuNum=GPUnum)
        test_dataset = ttdf.TED_Rating_depPOSonly_indices_Dataset(
            data_indices=test_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            wvec_index_maker=wvec,gpuNum=GPUnum)
    elif dataset_type == 'depposwordprosody':
        wvec=ttdf.wvec_index_maker(gpuNum=GPUnum)
        train_dataset = ttdf.TED_Rating_depPOSnorverbal_Dataset(
            data_indices=train_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            wvec_index_maker=wvec,gpuNum=GPUnum)
        test_dataset = ttdf.TED_Rating_depPOSnorverbal_Dataset(
            data_indices=test_id,firstThresh = firstThresh,
            secondThresh = secondThresh,scale_rating = scale_rating,
            wvec_index_maker=wvec,gpuNum=GPUnum)
    else:
        raise NotImplementedError(\
            'Only "word-only", "deppos", and "depposword" are supported dataset_type')
    
    # Length of the training and test (held out) dataset
    train_datalen = len(train_dataset)
    test_datalen = len(test_dataset)
    print 'Training Dataset Length:',train_datalen
    print 'Test Dataset Length:',test_datalen

    # Prepare the output folder and its contents
    outpath = os.path.join(ted_data_path,output_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # rand_filenum is a randomized element to make the filename of
    # different runs different. Note that the log file and the 
    rand_filenum = np.random.rand()
    output_log = \
    'LSTM_log_typ{0}_s{1}_{2}_{3}_hid{4}_it{5}_tr{6}_bat{7}_{8}.txt'.format(\
        dataset_type,scale_rating,firstThresh,secondThresh,\
        hidden_dim,max_iter_over_dataset,train_datalen,\
        minibatch_size,rand_filenum)
    outlogfile = os.path.join(outpath,output_log)
    model_outfile = output_log.replace('LSTM_log',\
        'LSTM_model').replace('.txt','.model')
    model_filename = os.path.join(outpath,model_outfile)

    # Build the model
    print 'Building the Neural Network Model ...'
    if dataset_type == 'word-only':    
        model = ttm.LSTM_TED_Rating_Predictor_wordonly(
            hidden_dim=hidden_dim,output_dim=len(train_dataset.ylabel),
            wvec_vals=train_dataset.wvec_map.w2v_vals,gpuNum=GPUnum,
            dropout=dropconnect)
    elif dataset_type == 'deppos':
        model = ttm.TreeLSTM(input_dim=hidden_dim,hidden_dim=hidden_dim,
            output_dim=len(train_dataset.ylabel),depidx=train_dataset.depidx,
            posidx=train_dataset.posidx,gpuNum=GPUnum,dropout=dropconnect)
    elif dataset_type == 'depposword':
        model = ttm.TreeLSTM(input_dim=16,hidden_dim=hidden_dim,
            output_dim=len(train_dataset.ylabel),depidx=train_dataset.depidx,
            posidx=train_dataset.posidx,includewords=True,gpuNum=GPUnum,
            dropout=dropconnect)
    elif dataset_type == 'depposwordprosody':
        model = ttm.TreeLSTM_with_Prosody(input_dim=16,hidden_dim=hidden_dim,
            output_dim=len(train_dataset.ylabel),depidx=train_dataset.depidx,
            posidx=train_dataset.posidx,includewords=True,gpuNum=GPUnum,
            dropout=dropconnect)
    else:
        raise IOError('Model type not recognized')
    print 'done'

    # Get loss function matching the model
    lossweight = float(secondThresh)/float(100. - secondThresh)
    loss_fn = model.loss_fn

    # Initialize the optimizer
    optimizer = optimizer_fn(model.parameters(),lr = learning_rate,weight_decay=weight_decay)

    # Preparing file to save the parameters and status
    with open(outlogfile,'wb') as fparam:    
        fparam.write('dataset_type={}'.format(dataset_type)+'\n')
        fparam.write('train_dataset_len={}'.format(train_datalen)+'\n')
        fparam.write('test_dataset_len={}'.format(test_datalen)+'\n')
        fparam.write('train_dataset_classname={}'.format(train_dataset.__class__.__name__)+'\n')
        fparam.write('test_dataset_classname={}'.format(test_dataset.__class__.__name__)+'\n')
        fparam.write('flatten_sentence={}'.format(flatten_sentence)+'\n')
        fparam.write('minibat_len={}'.format(minibatch_size)+'\n')
        fparam.write('scale_rating={}'.format(scale_rating)+'\n')
        fparam.write('firstThresh={}'.format(firstThresh)+'\n')
        fparam.write('secondThresh={}'.format(secondThresh)+'\n')
        fparam.write('hidden_dim={}'.format(hidden_dim)+'\n')
        fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
        fparam.write('learning_rate={}'.format(learning_rate)+'\n')
        fparam.write('weight_decay={}'.format(weight_decay)+'\n')
        fparam.write('dropconnect={}'.format(dropconnect)+'\n')
        fparam.write('model_outfile={}'.format(model_outfile)+'\n')
        fparam.write('modelclassname={}'.format(model.__class__.__name__)+'\n')
        fparam.write('modelclass={}'.format(str(model.__class__))+'\n')
        fparam.write('optimizerclassname={}'.format(optimizer.__class__.__name__)+'\n')
        fparam.write('optimizerclass={}'.format(str(optimizer.__class__))+'\n')        
        fparam.write('lossclassname={}'.format(loss_fn.__class__.__name__)+'\n')
        fparam.write('lossclass={}'.format(str(loss_fn.__class__))+'\n')        
        fparam.write('gpunum={}'.format(GPUnum)+'\n')
        fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
        fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')        

        # Multiple iteration over the training dataset
        min_test_loss = np.inf 
        old_time = time.time()
        for an_iter in range(max_iter_over_dataset):
            # Constructing minibaches. 
            # ------------------------
            # While constructing the minibatch, the
            # data also needs to be GPUtized. That is why it is time consuming.
            # For word-only dataset type, I tried to preload 
            # most of the information (word-vectors) in the GPU
            # and passed only the word-indices for faster loading.
            # I also tried several experiments to check faster loading option.
            # I still need to do some more experiments.
            # For now, the serial loading seems to be the fastest.
            # TODO: Implement for averaged and streamed dataset.
            ##################### DEBUG 1 ###############
            # oldtime = time.time()
            #############################################
            # Option 1: Fastest for now. Data is readily usable.
            minibatch_iter = ttdf.get_minibatch_iter(train_dataset,\
                minibatch_size)
            minibatch_iter_test = ttdf.get_minibatch_iter(test_dataset,\
                minibatch_size)
            # Option 2: Uses multiple processes, so, supposed to be fast.
            # But not. Creating and destroying multiple processes have large
            # overhead. Data is readily usable.
            # minibatch_iter = ttdf.get_minibatch_iter_pooled(train_dataset,\
            #    minibatch_size,GPUnum)
            # Option 3: Uses pytorch's Dataloader method. A bit faster. However,
            # the data is packed. This option is unusable because
            # packed data cannot be passed through the current model.
            #minibatch_iter = ttdf.get_data_iter_simple(train_dataset,\
            #    minibatch_size,GPUnum)
            # Option 4, TO DO: I need to experiment if I use the pytorch's
            # dataloader along with an unpacking the data to a usable form, does
            # it speedup the training?
            # Option 5, TO DO: If I do not preload the data into the GPU, and pass
            # the complete data through dataloader, followed by the pytorch's
            # implementation of LSTM (not LSTMCell), does it speedup the
            # training?
            ##################### DEBUG 2 ###############
            # print 'Time to construct minibach iterator:',time.time()-oldtime
            #############################################

            # ----------------------- Training Loop ---------------------------
            losslist=[]
            data_count = 0
            # Training time
            model.train()
            for i,minibatch in enumerate(minibatch_iter):
                model.zero_grad()
                # Forward pass through the model                
                log_probs = model(minibatch)

                # Calculate the loss
                loss = __compute_loss__(log_probs,minibatch,loss_fn,lossweight)
                # Backpropagation of the gradients
                lossval = loss.cpu().data.numpy()
                loss.backward()

                # Parameter update
                optimizer.step()

                # Logging the current status
                data_count += len(minibatch)
                ratio_trained = float(data_count)/float(train_datalen)
                losslist.append(lossval)
                status_msg='train:{0},Loss:{1:0.6},batch:{2},'
                status_msg+='Data_fed:{3:3.2}%,count:{4},iter_time:{5}'
                status_msg=status_msg.format(an_iter,lossval,i,\
                    ratio_trained*100,data_count,time.time()-old_time,'f')
                print status_msg
                fparam.write(status_msg + '\n')
            # Write the average loss of last iteration
            meanloss = np.nanmean(losslist)
            status_msg = 'Average Train loss:{0}'.format(meanloss)
            print status_msg
            print
            fparam.write(status_msg + '\n' + '\n')
            fparam.flush()
            os.fsync(fparam.fileno())

            # ----------------------- Test Loop ---------------------------
            losslist_test=[]
            data_count_test = 0                        
            model.eval()
            for i,minibatch_test in enumerate(minibatch_iter_test):
                # Grad calculation and optimizer is not required.
                with torch.no_grad():
                    # Forward pass through the model                
                    log_probs = model(minibatch_test)
                    # Calculate the loss
                    loss_test = __compute_loss__(log_probs,minibatch_test,loss_fn,lossweight)
                    lossval_test = loss_test.cpu().data.numpy()
            
                # Logging the current status
                data_count_test += len(minibatch_test)
                ratio_trained = float(data_count_test)/float(train_datalen)
                losslist_test.append(lossval_test)
                status_msg='test:{0},Loss:{1:0.6},batch:{2},'
                status_msg+='Data_fed:{3:3.2}%,count:{4},iter_time:{5}'
                status_msg=status_msg.format(an_iter,lossval_test,i,\
                    ratio_trained*100,data_count_test,time.time()-old_time,'f')
                print status_msg
                fparam.write(status_msg + '\n')
            # Write the average loss of last iteration
            meanloss = np.nanmean(losslist_test)
            status_msg = 'Average Test loss:{0}'.format(meanloss)
            print status_msg
            print
            fparam.write(status_msg + '\n' + '\n')
            fparam.flush()
            os.fsync(fparam.fileno())

            #  If the test loss decreases than the minimum test loss, 
            # save the model weights.
            if meanloss<min_test_loss:
                min_test_loss = meanloss
                model.save(open(model_filename,'wb'))

def resume_recurrent_training():
    '''
    Given an LSTM_log and corresponding LSTM_model file,
    resume the training/testing loop from last iteration.
    '''
    raise NotImplementedError

def __compute_loss__(log_probs,minibatch,loss_fn,lossweight):
    '''
    Compute the loss. It is important to make sure that loss function
    appropriately reflect the class distribution. That is, the false
    negatives should be weighted proportionally higher when the negative
    class has more data points than the positive class.
    '''
    losslist = []
    count=0.
    weights = torch.ones_like(log_probs[0])*lossweight
    loss_fn = loss_fn(pos_weight = weights)
    for i,an_item in enumerate(minibatch):        
        losslist.append(loss_fn(log_probs[i],an_item['Y']))
        count+=1.
    loss = reduce(torch.add,losslist)/count

    return loss


if __name__=='__main__':
    train_recurrent_models()

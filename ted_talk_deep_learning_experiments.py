import os
import time
import numpy as np

import torch.nn as nn
import torch.optim as optim

import sklearn as sl
import sklearn.metrics as met

import ted_talk_evaluate as ttev
import ted_talk_train as ttt
import ted_talk_results as ttr
import ted_talk_data_feeder as ttdf
from TED_data_location import ted_data_path

def exp0a_debug_train_SSE_small_data():
    '''
    Unit test code to train the deep learning model over a small data
    '''
    # ===== TRAIN =====
    start_time = time.time()
    # Build the model. Also try ttt.__build_RTE__
    model = ttt.__build_SSE__(reduced_val=True,sense_dim=14,gpunum=-1)
    # Train model
    ttt.train_model(model, ttdf.__tree_rating_feeder__,\
        output_folder = 'SSE_result/',max_data=2,max_iter=1)
    print 'Training time:',time.time() - start_time

def exp0b_debug_test_SSE_small_data():
    '''
    Unit test code to evaluate the deep learning model over a small data
    '''
    start_time = time.time()
    # Binarize the ratings for the whole dataset
    y_bin, label_names = ttdf.binarized_ratings()
    # Read the output log
    test_idx,_,model = ttr.read_output_log(result_dir = 'SSE_result/')
    # Prepare loss function
    loss_fn = nn.KLDivLoss(size_average=False)
    # Prepare the output file
    outfile = os.path.join(ted_data_path,'SSE_result/dev_result')
    # Evaluate the model
    ttev.evaluate_model(test_idx, model, loss_fn, data_feeder = \
        ttdf.__tree_rating_feeder__,y_gt_dict = y_bin, threshold = thresh,\
        y_labels=label_names,outfilename = outfile, max_data=2)
    print 'Evaluation time:',time.time() - start_time

def exp0c_debug_train_RTE_small_data():
    '''
    Unit test code to train the deep learning model (RTE) over a small data
    '''
    # ===== TRAIN =====
    start_time = time.time()
    # Build the model. Also try ttt.__build_RTE__
    model = ttt.__build_RTE__(reduced_val=True,sense_dim=14,gpunum=-1)
    # Train model
    ttt.train_model(model, ttdf.__tree_rating_feeder__,\
        output_folder = 'SSE_result/',max_data=2,max_iter=1)
    print 'Training time:',time.time() - start_time


def exp1_train(sense_dim = 14, outdir = 'run_0/', max_iter = 10,
    reduced_val = True,
    gpunum = -1,
    train_test_ratio = 0.80,
    loss_fn_name = nn.KLDivLoss,
    optim_fn_name = optim.Adam,
    learning_rate = 0.01,
    model_outfile = 'model_weights.pkl',
    output_log = 'train_logfile.txt',
    max_data = np.inf,
    model_initiator_fn = ttt.__build_RTE__):
    '''
    Unit test code to train the deep learning models with rating data
    Also try model_initiator_fn=ttt.__build_SSE__
    '''
    start_time = time.time()
    # Build the model
    model = model_initiator_fn(reduced_val,sense_dim=sense_dim,gpunum=gpunum)
    # Train model
    ttt.train_model(model, ttdf.__tree_rating_feeder__, outdir, 
        train_test_ratio, loss_fn_name, optim_fn_name, learning_rate,
        model_outfile, output_log, max_data, max_iter)
    print 'Training time:',time.time() - start_time


def exp2_evaluate(outdir,result_filename = 'dev_result',
    loss_fn_name = nn.KLDivLoss):
    '''
    Unit test code to evaluate the deep learning model with held out dataset
    '''
    start_time = time.time()
    # Prepare to evaluate
    y_bin, label_names = ttdf.binarized_ratings()
    test_idx,_,model = ttr.read_output_log(result_dir = outdir)
    loss_fn = loss_fn_name(size_average=False)
    outfile = os.path.join(os.path.join(ted_data_path,outdir),result_filename)
    # Evaluate the model
    results = ttev.evaluate_model(test_idx, model, loss_fn,\
        data_feeder = ttdf.__tree_rating_feeder__,\
        y_gt_dict = y_bin, threshold = thresh, y_labels=label_names,\
        outfilename = outfile)
    print 'Evaluation time:',time.time() - start_time


def exp3_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    taskID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    senselist = [2,8,12,14,18,20,22,25,28,30]
    print 'TaskID = ',taskID
    exp1_train(sense_dim=senselist[taskID],outdir='run_{}/'.format(taskID))
    print 'Training Done'
    exp2_evaluate(outdir='run_{}/'.format(taskID))
    print 'Evaluation Done'

# -------------------------------------------------------------

def exp4_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'word-only',
        firstThresh = 50.,
        secondThresh = 50.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 10,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.0033,
        weight_decay = 0.00066,
        max_iter_over_dataset = 48,
        GPUnum = 0)

def exp4_1_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'word-only',
        firstThresh = 50.,
        secondThresh = 50.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 10,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.00066,
        weight_decay = 0.00066,
        max_iter_over_dataset = 48,
        GPUnum = 0)

def exp4_2_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'word-only',
        firstThresh = 50.,
        secondThresh = 50.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 10,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.0001,
        weight_decay = 0.00066,
        max_iter_over_dataset = 48,
        GPUnum = 0)

def exp4_3_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'word-only',
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
        weight_decay = 0.00066,
        max_iter_over_dataset = 48,
        GPUnum = 0)

def exp5_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'word-only',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 10,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.0001,
        weight_decay = 0.0033,
        max_iter_over_dataset = 48,
        GPUnum = 0)


def exp6_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposword',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.00066,
        weight_decay = 0.00066,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = -1)    

def exp7_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposword',
        firstThresh = 50.,
        secondThresh = 50.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.00066,
        weight_decay = 0.00066,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = -1)    

def exp8_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'word-only',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.00066,
        weight_decay = 0.00066,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = 0)    

def exp9_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
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
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = 0) 

def exp10_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposwordprosody',
        firstThresh = 50.,
        secondThresh = 50.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 10,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adagrad,
        learning_rate = 0.01,
        weight_decay = 0.00033,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = 0) 

def exp11_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposwordprosody',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.001,
        weight_decay = 0.00033,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = 0) 

def exp12_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposwordprosody',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adagrad,
        learning_rate = 0.01,
        weight_decay = 0.00033,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = 0) 

def exp13_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposwordprosody',
        firstThresh = 50.,
        secondThresh = 50.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 10,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adagrad,
        learning_rate = 0.01,
        weight_decay = 0.0001,
        dropconnect=0.2,
        max_iter_over_dataset = 70,
        GPUnum = -1) 

def exp14_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposwordprosody',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adam,
        learning_rate = 0.001,
        weight_decay = 0.00033,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = -1) 

def exp15_run_in_Bluehive():
    '''
    Run the training and test module of the deep learning model with 
    appropriate parameters in Bluehive
    '''
    ttt.train_recurrent_models(
        dataset_type = 'depposwordprosody',
        firstThresh = 30.,
        secondThresh = 70.,
        scale_rating = True,
        flatten_sentence = False,
        minibatch_size = 30,
        hidden_dim = 128,
        output_folder = 'TED_models/',
        train_test_ratio = 0.90,
        optimizer_fn = optim.Adagrad,
        learning_rate = 0.01,
        weight_decay = 0.00033,
        dropconnect=0.2,
        max_iter_over_dataset = 48,
        GPUnum = -1) 

if __name__=='__main__':
    # Control from here which experiment is going to run
    exp6_run_in_Bluehive()
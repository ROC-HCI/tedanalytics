import os
import time
import numpy as np

import torch.nn as nn
import torch.optim as optim

import sklearn as sl
import sklearn.metrics as met

import ted_talk_evaluate as tte
import ted_talk_train as ttt
import ted_talk_train_results as tttr
import ted_talk_data_feeder as ttdf

def exp0_debug_train_test_SSE_small_data():
    '''
    Unit test code to train and evaluate the model over a small data
    '''
    # ===== TRAIN =====
    start_time = time.time()
    # Build the model. Also try ttt.__build_RTE__
    model = ttt.__build_SSE__(reduced_val=True,sense_dim=14,gpunum=-1)
    # Train model
    ttt.train_model(model, ttdf.__tree_rating_feeder__,\
        output_folder = 'SSE_result/',max_data=10)
    print 'Training time:',time.time() - start_time
    
    # ===== TEST =====
    start_time = time.time()
    # Binarize the ratings for the whole dataset
    y_bin, thresh, label_names = ttdf.binarized_ratings()
    # Read the output log
    test_idx,_,model = tttr.read_output_log(result_dir = 'SSE_result/')
    # Prepare loss function
    loss_fn = nn.KLDivLoss(size_average=False)
    # Prepare the output file
    outfile = os.path.join(ted_data_path,'SSE_result/dev_result')
    # Evaluate the model
    tte.evaluate_model(test_idx, model, loss_fn, data_feeder = \
        ttdf.__tree_rating_feeder__,y_gt_dict = y_bin, threshold = thresh,\
        y_labels=label_names,outfilename = outfile, max_data=5)
    print 'Evaluation time:',time.time() - start_time

def exp1_train(sense_dim=14, outdir='run_0/', max_iter = 10,
    reduced_val=True,
    gpunum = -1,
    train_test_ratio = 0.80,
    loss_fn_name = nn.KLDivLoss,
    optim_fn_name = optim.Adam,
    learning_rate = 0.01,
    model_outfile = 'model_weights.pkl',
    output_log = 'train_logfile.txt',
    max_data = np.inf,
    model_initiator_fn=ttt.__build_RTE__):
    '''
    Unit test code to train the model with rating data
    Also try model_initiator_fn=ttt.__build_SSE__
    '''
    start_time = time.time()
    # Build the model
    model = model_initiator_fn(reduced_val,sense_dim=sense_dim,gpunum=gpunum)
    # Train model
    train_model(model, ttdf.__tree_rating_feeder__, outdir, train_test_ratio,
        loss_fn_name, optim_fn_name, learning_rate, model_outfile,
        output_log, max_data, max_iter)
    print 'Training time:',time.time() - start_time


def exp2_evaluate(outdir,result_filename='dev_result',
    loss_fn_name = nn.KLDivLoss):
    '''
    Unit test code to evaluate the model with held out dataset
    '''
    start_time = time.time()
    # Prepare to evaluate
    y_bin, thresh, label_names = ttdf.binarized_ratings()
    test_idx,_,model = tttr.read_output_log(result_dir = outdir)
    loss_fn = loss_fn_name(size_average=False)
    outfile = os.path.join(os.path.join(ted_data_path,outdir),result_filename)
    # Evaluate the model
    results = evaluate_model(test_idx, model, loss_fn,\
        data_feeder = ttdf.__tree_rating_feeder__,\
        y_gt_dict = y_bin, threshold = thresh, y_labels=label_names,\
        outfilename = outfile)
    print 'Evaluation time:',time.time() - start_time


def exp3_run_in_Bluehive():
    '''
    Run the training and test module with appropriate parameters in Bluehive
    '''
    taskID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    senselist = [2,8,12,14,18,20,22,25,28,30]
    print 'TaskID = ',taskID
    exp1_train(sense_dim=senselist[taskID],outdir='run_{}/'.format(taskID))
    print 'Training Done'
    exp2_evaluate(outdir='run_{}/'.format(taskID))
    print 'Evaluation Done'
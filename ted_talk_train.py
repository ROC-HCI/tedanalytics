import os
import time
import json
import glob
import numpy as np
import cPickle as cp
import sklearn as sl
import sklearn.metrics as met
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
    Initiate a Syntactic-Semantic-Engine and initiates with data.
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
    Initiate a Syntactic-Semantic-Engine and initiates with data.
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


#def __revised_TreeEncoder__(reduced_val,sense_dim=14,gpunum=-1,\
#    final_activation=F.log_softmax):
#    '''
#    Initiate a Syntactic-Semantic-Engine and initiates with data.
#    If reduced, the output of each individual dependency tree
#    is averaged and then the final activation function is
#    applied. 
#    '''
#    # Reading Vocabs
#    print 'Reading Vocabs'
#    _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
#    wvec = ttdf.read_crop_glove()
#    # Initialize the model
#    model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,wvec,\
#        reduced=reduced_val,GPUnum=gpunum,sensedim=sense_dim,\
#        final_activation=final_activation)
#    print 'Model Initialized'
#    return model


def __tree_rating_feeder__(atalk,gpunum=-1):
    '''
    Transforms the data and the ground truth in appropriate format so that
    it can be used as a feeder-function argument for train_model function.
    '''
    # All the dependency trees and the rating
    all_deptree,y,_ = ttdf.get_dep_rating(atalk)
    # Construct ground truth tensor
    if gpunum < 0:
        rating_t = autograd.Variable(torch.FloatTensor(y))
    else:
        with torch.cuda.device(gpunum):
            rating_t = autograd.Variable(torch.cuda.FloatTensor(y))
    return all_deptree,rating_t.view(1,-1)

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
                # Save the loss in last iteration for computing average
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

def read_output_log(result_dir = 'SSE_result/',logfile = 'train_logfile.txt'):
    '''
    Given a output folder containing the log and model file
    (generated by the train_model function), this function reads the
    log and returns the test indices and the model.
    It also plots the losses as an added convenience.
    '''
    inpath = os.path.join(ted_data_path,result_dir)
    logpath = os.path.join(inpath,logfile)
    # Open the log file
    with open(logpath) as fin:
        alllosses=[]
        loss_index = -1
        # Read the log file
        for aline in fin:
            if aline.startswith('test_indices'):
                # Reading test indices
                test_idx = json.loads(aline.split('=')[1])
            elif aline.startswith('train_indices'):
                # Skip printing the training indices. TL;
                continue
            elif aline.startswith('training:'):
                # Enlist losses
                for items in aline.strip().split(','):
                    elems = items.split(':')
                    if elems[0].strip()=='Loss':
                        alllosses.append(float(elems[1].strip()))
            elif aline.startswith('model_outfile'):
                # Find the model file
                modelfile = aline.strip().split('=')[1]
            else:
                print aline.strip()
        # Save a plot of the loss
        __plot_losses__(alllosses,os.path.join(inpath,'losses.eps'))
        # Load the trained model
        model = torch.load(os.path.join(inpath,modelfile))
        return test_idx, model

def evaluate_model(test_idx, model, loss_fn, data_feeder, \
    y_gt_dict, threshold, y_labels, outfilename, max_data = np.inf):
    '''
    Evaluate a trained model with the held out dataset.
    Most of the inputs to this function can be obtained from the
    ted_talk_data_feeder.binarized_ratings function. The model argument takes
    the pretrained model that is under evaluation.
    This function returns the results of evaluation (e.g. precision, recall,
    accuracy, f1 score etc.) as well as the average loss over the held out data.
    The results are returned as a dictionary. This dictionary is also saved
    as a pickle file. outfilename should contain the full path of this result
    file. max_data specifies how many of the held out data (test_idx) will be
    used for actual evaluation.
    '''
    test_losses=[]
    result_map = {}
    y_gt = []
    y_test = []
    y_test_score = []
    print 'Model evaluation:'
    for i,atalk in enumerate(test_idx):
        if i >= max_data:
            break
        # Get the dependency trees and the rating
        all_deptree,rating_t = data_feeder(atalk)
        # Forward pass through the model and preserve
        log_probs = model(all_deptree)
        y_temp = np.exp(log_probs.data[0].numpy())
        # Preserve proba
        y_test_score.append(y_temp.tolist())
        # Binarize and preserve
        y_test.append([-1. if y<=t else 1. for y,t in zip(y_temp,threshold)])
        # Calculate the loss and preserve
        loss = loss_fn(log_probs,rating_t)
        lossval = loss.data[0]
        test_losses.append(lossval)
        # Preserve Ground Truth
        y_gt.append(y_gt_dict[atalk])
        # Show status
        print 'id:',atalk,'remaining:',len(test_idx)-(i+1),'loss:',lossval
    # Show average test loss
    average_loss = np.mean(test_losses)
    print 'Average Loss:',
    # Perform evaluation
    results = __classifier_eval__(y_gt,y_test,y_test_score,y_labels,outfilename)
    # Add the average loss
    results['average_loss'] = average_loss
    # Save results in pkl file
    cp.dump(results,open(outfilename+'.pkl','wb'))
    return results

def combine_results(resultfilename,folder_prefix='run_'):
    '''
    This function reads the pickle files containing model evaluation results
    (obtained by executing evaluate_model function) and computes the average.
    These pickle files are assumed to be located in different folders starting
    with the same prefix in the ted_data_path. 
    '''
    infolders = glob.glob(os.path.join(ted_data_path,folder_prefix)+'*')
    combined={}
    for afolder in infolders:
        results = cp.load(open(os.path.join(afolder,resultfilename)))
        for akey in results:
            if results[akey] and akey!='order':
                if not akey in combined:
                    combined[akey] = [results[akey]]
                else:
                    combined[akey].append(results[akey])
    for akey in combined:
        combined[akey] = np.mean(combined[akey],axis=0).tolist()
    combined['order']=results['order']
    return combined

def error_analysis(resultfile='dev_result.pkl',\
    train_log='train_logfile.txt',folder_prefix='run_',outfile='error_analysis.eps'):
    '''
    This function reads the model evaluation results (obtained by executing 
    evaluate_model function) and plots the losses with respect to model
    parameters (sense_dim).
    The results are assumed to be located in different folders starting
    with the same prefix in the ted_data_path.
    '''
    infolders = glob.glob(os.path.join(ted_data_path,folder_prefix)+'*')
    outfilename = os.path.join(ted_data_path,outfile)
    senselist=[]
    trainlosslist=[]
    testlosslist=[]
    for afolder in infolders:
        logfilename = os.path.join(afolder,train_log)
        currentresultfile = os.path.join(afolder,resultfile)
        # Read the training log
        with open(logfilename) as fin:
            for aline in fin:
                if aline.startswith('sense_dim'):
                    senselist.append(int(aline.strip().split('=')[1]))
                if aline.startswith('Average Loss in last iteration'):
                    trainlosslist.append(float(aline.strip().split(':')[1]))
        # Read the current result file
        results = cp.load(open(currentresultfile))
        testlosslist.append(results['average_loss'])
    # Sort results
    idx = np.argsort(senselist)
    senselist,trainlosslist,testlosslist = zip(*[(senselist[i],\
            trainlosslist[i],testlosslist[i]) for i in idx])

    # Plot the numbers
    plt.figure(1)
    plt.clf()
    plt.plot(senselist,trainlosslist,color='blue',label='Train Loss')
    plt.plot(senselist,testlosslist,color='red',label='Test Loss')
    plt.xlabel('Sense Vector Length (Model Complexity)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outfilename)
    plt.close()    
    return senselist,trainlosslist,testlosslist

def __classifier_eval__(y_gt,y_test,y_test_score,col_labels,\
    outfilename='',bypass_ROC=True):
    '''
    Helper function to return classification results. Returns a dictionary of dictionaries
    reporting the precision, recall, accuracy, f1 score and AUC for each column of y.
    '''
    if type(y_gt)==list:
        y_gt = np.array(y_gt)
    if type(y_test)==list:
        y_test = np.array(y_test)
    if type(y_test_score)==list:
        y_test_score = np.array(y_test_score)
    results = {}
    # loop through every column of y (different types of ratings)
    for col in range(len(col_labels)):
        # Checking if data for both classes is available
        gt_unq = np.unique(y_gt[:,col])
        if gt_unq.shape[0] <= 1:
            print col_labels[col]
            print 'Data sample contains one or less class ... skipping'
            results[col_labels[col]]={}
            continue
        # Calculating results
        prec,rec,fscore, support = met.precision_recall_fscore_support(\
            y_gt[:,col],y_test[:,col])
        auc = met.roc_auc_score(y_gt[:,col],y_test_score[:,col])
        accur = met.accuracy_score(y_gt[:,col],y_test[:,col])
        results[col_labels[col]] = [np.mean(prec),np.mean(rec),\
            np.mean(fscore),accur,auc]
        # Show report
        print col_labels[col]
        print met.classification_report(y_gt[:,col],y_test[:,col])
        print 'Accuracy:',accur
        print 'AUC:',auc
        # Draw the ROC curve if requested        
        if not bypass_ROC:
            fpr,tpr,_ = met.roc_curve(y_gt[:,col],y_test_score[:,col])            
            # Plot ROC Curve
            plt.figure(0)
            plt.clf()
            plt.plot(fpr,tpr,color='darkorange',\
                label='ROC Curve (AUC={0:0.2f})'.format(auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig(outfilename+'_'+col_labels[col]+'.eps')
            plt.close()
    results['order']=['avg_precision','avg_recall','avg_fscore','accuracy','auc']
    return results

def __plot_losses__(alllosses,dest):
    # Plots the losses
    plt.figure(1)
    plt.clf()
    plt.semilogy(alllosses)
    plt.xlabel('Iterations (1 sample per iteration)')
    plt.ylabel('Loss')
    plt.savefig(dest)
    plt.close()

# ----------------------------- Unit Test Codes -------------------------------

def exp0_debug_train_test_SSE_small_data():
    '''
    Unit test code to train and evaluate the model over a small data
    '''
    # ===== TRAIN =====
    start_time = time.time()
    # Build the model
    model = __build_SSE__(reduced_val=True,sense_dim=14,gpunum=-1)
    # Train model
    train_model(model, __tree_rating_feeder__,\
        output_folder = 'SSE_result/',max_data=10)
    print 'Training time:',time.time() - start_time
    
    # ===== TEST =====
    start_time = time.time()
    # Binarize the ratings for the whole dataset
    y_bin, thresh, label_names = ttdf.binarized_ratings()
    # Read the output log
    test_idx, model = read_output_log(result_dir = 'SSE_result/')
    # Prepare loss function
    loss_fn = nn.KLDivLoss(size_average=False)
    # Prepare the output file
    outfile = os.path.join(ted_data_path,'SSE_result/dev_result')
    # Evaluate the model
    evaluate_model(test_idx, model, loss_fn, data_feeder = __tree_rating_feeder__,\
        y_gt_dict = y_bin, threshold = thresh, y_labels=label_names,\
        outfilename = outfile, max_data=5)
    print 'Evaluation time:',time.time() - start_time

def exp1_train_SSE(sense_dim=14, outdir='run_0/', max_iter = 10,
    reduced_val=True,
    gpunum = -1,
    train_test_ratio = 0.80,
    loss_fn_name = nn.KLDivLoss,
    optim_fn_name = optim.Adam,
    learning_rate = 0.01,
    model_outfile = 'model_weights.pkl',
    output_log = 'train_logfile.txt',
    max_data = np.inf,
    model_initiator_fn=__build_SSE__):
    '''
    Unit test code to train the model with rating data
    '''
    start_time = time.time()
    # Build the model
    model = model_initiator_fn(reduced_val,sense_dim=sense_dim,gpunum=gpunum)
    # Train model
    train_model(model, __tree_rating_feeder__, outdir, train_test_ratio,
        loss_fn_name, optim_fn_name, learning_rate, model_outfile,
        output_log, max_data, max_iter)
    print 'Training time:',time.time() - start_time


def exp2_evaluate_SSE(outdir,result_filename='dev_result',
    loss_fn_name = nn.KLDivLoss):
    '''
    Unit test code to evaluate the model with held out dataset
    '''
    start_time = time.time()
    # Prepare to evaluate
    y_bin, thresh, label_names = ttdf.binarized_ratings()
    test_idx, model = read_output_log(result_dir = outdir)
    loss_fn = loss_fn_name(size_average=False)
    outfile = os.path.join(os.path.join(ted_data_path,outdir),result_filename)
    # Evaluate the model
    results = evaluate_model(test_idx, model, loss_fn,\
        data_feeder = __tree_rating_feeder__,\
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
    exp1_train_SSE(sense_dim=senselist[taskID],outdir='run_{}/'.format(taskID))
    print 'Training Done'
    exp2_evaluate_SSE(outdir='run_{}/'.format(taskID))
    print 'Evaluation Done'



if __name__=='__main__':
    exp3_run_in_Bluehive()


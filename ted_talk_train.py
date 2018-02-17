import os
import time
import json
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

# ========= Development Temporarily stalled ======================
# def train_humor_sentencewise(output_folder = 'SSE_result/',
#     sense_dim = 2,
#     train_test_ratio = 0.75,
#     final_activation = F.log_softmax,
#     model_outfile = 'model_weights.pickle',
#     output_log = 'logfile.txt',
#     gpunum = -1,
#     max_data=np.inf,
#     max_iter = 3):
#     '''
#     Procedure to train the SSE with sentences followed by laughter or 
#     non-laughter tags.
#     '''    
#     outpath = os.path.join(ted_data_path,output_folder)
#     if not os.path.exists(outpath):
#         os.makedirs(outpath)

#     # Prepare trining and test split
#     train_id,test_id = ttdf.split_train_test(train_test_ratio)
#     np.random.shuffle(train_id)
#     np.random.shuffle(test_id)

#     # Reading Vocabs
#     print 'Reading Vocabs'
#     _,dep_dict,_,pos_dict = ttdf.read_dep_pos_vocab()
#     wvec = ttdf.read_crop_glove()
#     # Initialize the model
#     model = ttm.SyntacticSemanticEngine(dep_dict,pos_dict,wvec,\
#         GPUnum=gpunum,sensedim=sense_dim,final_activation=final_activation)
#     # Initialize the optimizer
#     optimizer = optim.Adam(model.parameters(),lr = 0.01)
#     # Use Negative Log Likelihood Loss
#     loss_fn = nn.NLLLoss(size_average=False)
#     # Show status
#     print 'Model Loaded'
    
#     # Save the parameters of the function call. It allows me to 
      # audit the models
#     with open(os.path.join(outpath,output_log),'wb') as fparam:
#         fparam.write('sense_dim={}'.format(sense_dim)+'\n')
#         fparam.write('train_test_ratio={}'.format(train_test_ratio)+'\n')
#         fparam.write('activation={}'.format(model.activation.__repr__())+'\n')
#         fparam.write('final_activation={}'.format(\
#             model.final_activation.__repr__())+'\n')
#         fparam.write('model_outfile={}'.format(model_outfile)+'\n')
#         fparam.write('gpunum={}'.format(gpunum)+'\n')
#         fparam.write('Optimizer_name={}'.format(optimizer.__repr__())+'\n')
#         fparam.write('Loss_name={}'.format(loss_fn.__repr__())+'\n')
#         fparam.write('train_indices={}'.format(json.dumps(train_id))+'\n')
#         fparam.write('test_indices={}'.format(json.dumps(test_id))+'\n')
        
#         for iter in range(max_iter):
#             np.random.shuffle(train_id)
#             # Write the loss in file
#             for i,atalk in enumerate(train_id):
#                 if i > max_data:
#                     break
#                 # Feed the whole talk as a minibatch
#                 minibatch = [(adep,label) for adep,label,_,_ in \
#                     ttdf.generate_dep_tag(atalk)]
#                 X,y = zip(*minibatch)
#                 # Construct the label tensor
#                 if gpunum < 0:
#                     labels_t = autograd.Variable(torch.LongTensor(y))
#                 else:
#                     with torch.cuda.device(gpunum):
#                         labels_t = autograd.Variable(torch.cuda.LongTensor(y))
                
#                 # Clear gradients from previous iterations
#                 model.zero_grad()
#                 # Forward pass through the model
#                 log_probs = model(X)
#                 # Calculate the loss
#                 loss = loss_fn(log_probs,labels_t)
#                 # Backpropagation of the gradients
#                 loss.backward()
#                 # Parameter update
#                 optimizer.step()

#                 # Show status
#                 lossval = loss.data[0]
#                 status_msg =  'training:'+str(atalk)+', Loss:'+str(lossval)+\
#                     ' #of_trees:'+str(len(y))+' avg_loss_per_tree:'+\
#                     str(lossval/len(y))
#                 print status_msg
#                 fparam.write(status_msg + '\n')
#     # Save the model
#     model_filename = os.path.join(outpath,model_outfile)
#     torch.save(model.cpu(),open(model_filename,'wb'))

def __build_SSE__(reduced_val,sense_dim,gpunum=-1,\
    final_activation=F.log_softmax):
    '''
    Initiate a Syntactic-Semantic-Engine and initiates with data
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

def __rating_feeder__(atalk,gpunum=-1):
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
    train_test_ratio = 0.75,
    loss_fn_name = nn.KLDivLoss,
    opt_fn_name = optim.Adam,
    learning_rate = 0.01,
    model_outfile = 'model_weights.pkl',
    output_log = 'logfile.txt',
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
    optimizer = opt_fn_name(model.parameters(),lr = learning_rate)

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

                #TODO: Calculate classifier performance and confusion matrix
                # for the training data
                
                # Logging the current status
                lossval = loss.data[0]
                status_msg =  'training:'+str(atalk)+', Loss:'+\
                    str(lossval)+', iteration:'+str(iter)
                print status_msg
                fparam.write(status_msg + '\n')
    # Save the model
    model_filename = os.path.join(outpath,model_outfile)
    torch.save(model.cpu(),open(model_filename,'wb'))

def read_output_log(result_dir = 'SSE_result/',logfile = 'logfile.txt'):
    '''
    Given a output folder containing log, model and misc file
    (output of the trainer function), this function reads the
    log and returns the training indices and model.
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
                print aline.strip()
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
    Evaluate a trained model with the held out data.
    The input to this function can be obtained from read_output_log
    and ted_talk_data_feeder.binarized_ratings.
    '''
    test_losses=[]
    result_map = {}
    y_gt = []
    y_test = []
    y_test_score = []
    for i,atalk in enumerate(test_idx):
        if i >= max_data:
            break
        # Get the dependency trees and the rating
        all_deptree,rating_t = data_feeder(atalk)
        # Forward pass through the model and preserve
        log_probs = model(all_deptree)
        y_temp = np.exp(log_probs.data[0].numpy())
        # Preserve proba
        y_test_score.append(y_temp)
        # Binarize and preserve
        y_test.append([-1. if y<=t else 1. for y,t in zip(y_temp,threshold)])            
        # Calculate the loss and preserve
        loss = loss_fn(log_probs,rating_t)
        lossval = loss.data[0]
        test_losses.append(lossval)
        # Preserve Ground Truth
        y_gt.append(y_gt_dict[atalk])
        # Show status
        print 'Done. id:',atalk,'remaining:',len(test_idx)-(i+1),\
            'test loss:',lossval
    # Show average test loss
    print 'Average Test Loss:',np.mean(test_losses)
    # Perform evaluation
    __classifier_eval__(y_gt,y_test,y_test_score,y_labels,outfilename)

    return y_test,y_test_score,y_gt,test_idx[:max_data]


def __classifier_eval__(y_gt,y_test,y_test_score,y_labels,outfilename,ROCTitle=None):
    '''
    Helper function to show classification results. Produces classification
    reports, accuracy and ROC curve.
    '''
    y_gt = np.array(y_gt)
    y_test = np.array(y_test)
    y_test_score = np.array(y_test_score)

    for col in range(len(y_labels)):
        print col,y_labels[col],y_gt.shape,y_test.shape
        # Checking for error
        gt_unq = np.unique(y_gt[:,col]).shape[0]
        tst_unq = np.unique(y_test[:,col]).shape[0]
        if not gt_unq == tst_unq == 2:
            print 'data sample does not contain both classes ... skipping'
            continue
        # Printing Report
        print sl.metrics.classification_report(y_gt[:,col],y_test[:,col])
        print 'Accuracy:',sl.metrics.accuracy_score(y_gt[:,col],y_test[:,col])
        auc = met.roc_auc_score(y_gt[:,col],y_test_score[:,col])
        print 'AUC:',auc
        fpr,tpr,_ = sl.metrics.roc_curve(y_gt[:,col],y_test_score[:,col],pos_label=1)
        plt.figure(0)
        plt.clf()
        plt.plot(fpr,tpr,color='darkorange',label='ROC Curve (AUC={0:0.2f})'.\
            format(auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if ROCTitle:
            plt.title(ROCTitle)
        plt.legend()
        plt.savefig(outfilename+'.eps')
        plt.close()


def __plot_losses__(alllosses,dest):
    # Plot the losses
    plt.close()
    plt.figure(1)
    plt.semilogy(alllosses)
    plt.xlabel('Iterations (1 sample per iteration)')
    plt.ylabel('Loss')
    plt.savefig(dest)

if __name__=='__main__':
    # start_time = time.time()
    # model = __build_SSE__(reduced_val=True,sense_dim=14,gpunum=-1)
    # #train_humor_sentencewise(gpunum=-1,max_data=10)
    # train_model(model, __rating_feeder__,max_data=10)
    # print 'time:',time.time() - start_time
    # Evaluate the model
    start_time = time.time()
    # Binarize the ratings for the whole dataset
    y_bin, thresh, label_names = ttdf.binarized_ratings()
    # Read the output log
    test_idx, model = read_output_log(result_dir = 'SSE_result/')
    # Prepare loss function
    loss_fn = nn.KLDivLoss(size_average=False)
    # Evaluate the model
    outfile = os.path.join(ted_data_path,'SSE_result/classifier_ROC')
    evaluate_model(test_idx, model, loss_fn, data_feeder = __rating_feeder__,\
        y_gt_dict = y_bin, threshold = thresh, y_labels=label_names,\
        outfilename = outfile, max_data=5)

    print 'time:',time.time() - start_time




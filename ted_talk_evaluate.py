import glob
import os
import json
import numpy as np
import cPickle as cp
import sklearn.metrics as met
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import ted_talk_data_feeder as ttdf
import ted_talk_results as ttr
import ted_talk_models as ttm
import list_of_talks



def evaluate_model(test_idx, model, loss_fn, data_feeder, \
    y_gt_dict, threshold, y_labels, outfilename, max_data = np.inf):
    '''
    This function evaluates only trained SyntacticSemanticEngine and
    RevisedTreeEncoder models with held out dataset.
    Please note that the FINAL test should run on a hidden test set indices.
    In that case, the test_idx should equal list_of_talks.test_set.

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

def __evaluate_all_models__():
    for alog in glob.glob('/scratch/mtanveer/TED_models/*.txt'):
        print alog                                                                  
        outname = alog.replace('LSTM_log','LSTM_results_dev').replace('.txt','.pkl')
        # check if output exists          
        outname = os.path.join('/scratch/mtanveer/TED_stats/Backup_results/',os.path.split(outname)[-1])
        print 'output:',outname            
        if os.path.exists(outname):        
            print 'file already processed. skipping ...'
            continue
        evaluate_recurrent_models(alog)

def __clean_existing_pkl__():
    for alog in glob.glob('/scratch/mtanveer/TED_models/*.txt'):
        outname = alog.replace('LSTM_log','LSTM_results_dev').replace('.txt','.pkl')
        # check if output exists          
        outname = os.path.join('/scratch/mtanveer/TED_stats/Backup_results/',os.path.split(outname)[-1])
        if os.path.exists(outname):        
            print 'file already exists. Moving the corresponding file from TED_models'
            outname_id = outname.split('.')[-2]
            for afile in glob.glob('/scratch/mtanveer/TED_models/*'+outname_id+'*'):
                oldpath,filename = os.path.split(afile)
                newfile = os.path.join(\
                    '/scratch/mtanveer/TED_stats/Backup_results/',filename)
                print afile,
                print 'moved to:',newfile
                os.rename(afile,newfile)

def evaluate_recurrent_models(logfilename,test_id=list_of_talks.test_set):
    '''
    Evaluation code for LSTM models that take sequential datasets.
    '''
    modelfilename = logfilename.replace('LSTM_log','LSTM_model').replace('.txt','.model')
    if not os.path.exists(modelfilename):
        raise IOError('{} does not exist'.format(modelfilename))
    # create output file
    output1 = logfilename.replace('LSTM_log','LSTM_results_dev').replace('.txt','.pkl')
    output2 = logfilename.replace('LSTM_log','LSTM_results_finaltest').replace('.txt','.pkl')
    # Load log data and model    
    logdata = ttr.read_lstm_log(logfilename)

    # Model will be loaded in the cpu/GPU according to the log
    model = ttm.load_model(modelfilename,logdata['modelclassname'])
    ttdf.gputize(model,model.gpuNum)
    model.eval()

    #--------------- Evaluate using the held-out dataset ------------------
    print 'Results with Dev dataset:'
    print '========================='
    logdata['test_indices'] = json.loads(logdata['test_indices'])
    results1 = get_results(logdata,model,logdata['test_indices'])
    results1.update(logdata)
    cp.dump(results1,open(output1,'wb'))
    #---------------- Evaluate using the final test dataset ----------------
    print 'Results with held-out final test dataset:'
    print '========================================='
    results2 = get_results(logdata,model,test_id)
    logdata['test_indices']=test_id
    results2.update(logdata)
    cp.dump(results2,open(output2,'wb'))
    print 'results saved in:'
    print output1
    print output2

def get_results(logdata,model,test_indices):
    if logdata['dataset_type'] == 'word-only':
        test_dataset = ttdf.TED_Rating_wordonly_indices_Dataset(
            data_indices = test_indices,
            firstThresh = logdata['firstThresh'],
            secondThresh = logdata['secondThresh'],
            scale_rating = bool(logdata['scale_rating']),
            flatten_sentence=False,
            access_hidden_test=list_of_talks.test_set==test_indices)    
    elif logdata['dataset_type'] == 'deppos':
        test_dataset = ttdf.TED_Rating_depPOSonly_indices_Dataset(
            data_indices = test_indices,
            firstThresh = logdata['firstThresh'],
            secondThresh = logdata['secondThresh'],
            scale_rating = bool(logdata['scale_rating']),
            flatten_sentence=False,
            access_hidden_test=list_of_talks.test_set==test_indices,
            gpuNum=model.gpuNum)
    elif logdata['dataset_type'] == 'depposword':
        wvec = ttdf.wvec_index_maker(model.gpuNum)
        test_dataset = ttdf.TED_Rating_depPOSonly_indices_Dataset(
            data_indices = test_indices,
            firstThresh = logdata['firstThresh'],
            secondThresh = logdata['secondThresh'],
            scale_rating = bool(logdata['scale_rating']),
            flatten_sentence=False,
            access_hidden_test=list_of_talks.test_set==test_indices,
            wvec_index_maker=wvec,
            gpuNum=model.gpuNum)
    elif logdata['dataset_type'] == 'depposwordprosody':
        wvec = ttdf.wvec_index_maker(model.gpuNum)
        test_dataset = ttdf.TED_Rating_depPOSnorverbal_Dataset(
            data_indices = test_indices,
            firstThresh = logdata['firstThresh'],
            secondThresh = logdata['secondThresh'],
            scale_rating = bool(logdata['scale_rating']),
            flatten_sentence=False,
            access_hidden_test=list_of_talks.test_set==test_indices,
            wvec_index_maker=wvec,
            gpuNum=model.gpuNum)
    else:
        raise NotImplementedError(
        'Only "word-only", "deppos", and "depposword" are supported')
    
    # Load minibatch
    minibatch_iter = ttdf.get_minibatch_iter(test_dataset,20)
    
    # get model predictions
    y_score=[]
    y_gt=[]
    y_predict=[]
    test_losses=[]
    for i,minibatch in enumerate(minibatch_iter):
        # Get prediction probability from the model
        with torch.no_grad():
            model_output = model(minibatch)
            if logdata['lossclassname']=='BCEWithLogitsLoss':
                # BCEWithLogitsLoss applies the sigmoid within the loss function
                try:
                    temp = map(lambda x:F.sigmoid(x.squeeze()).numpy().tolist(),\
                        model_output)
                except TypeError:
                    temp = map(lambda x:F.sigmoid(x.squeeze()).cpu().numpy().tolist(),\
                        model_output)
            else:
                try:
                    temp = map(lambda x:x.squeeze().numpy().tolist(),model_output)
                except TypeError:
                    temp = map(lambda x:x.squeeze().cpu().numpy().tolist(),model_output)

        pred_threshold = 0.5
        pred = map(lambda x:[float(an_x>pred_threshold) for an_x in x],temp)
        try:
            a_gt = [an_item['Y'].numpy().squeeze().tolist() for an_item in minibatch]
        except TypeError:
            a_gt = [an_item['Y'].cpu().numpy().squeeze().tolist() for an_item in minibatch]
        y_score.extend(temp)
        y_predict.extend(pred)
        y_gt.extend(a_gt)
        print 'processed:',(i+1)*20,'out of',len(test_dataset)
    results = __classifier_eval__(y_gt,y_predict,y_score,test_dataset.ylabel)    
    return results

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
            results[col_labels[col]]=[]
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
            plt.savefig(outfilename+'_'+col_labels[col]+'.png')
            plt.close()
    results['order']=['avg_precision','avg_recall','avg_fscore','accuracy','auc']
    return results

if __name__=='__main__':
    __evaluate_all_models__()
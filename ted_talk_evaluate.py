import numpy as np
import cPickle as cp
import sklearn.metrics as met
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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
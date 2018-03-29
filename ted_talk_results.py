import os
import json
import glob
import csv
import numpy as np
import cPickle as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from TED_data_location import ted_data_path


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
                test_idx = json.loads(aline.strip().split('=')[1])
            elif aline.startswith('train_indices'):
                train_idx = json.loads(aline.strip().split('=')[1])
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
        dest = os.path.join(inpath,'losses.png')
        # Plot the losses vs. iteration
        plt.figure(1)
        plt.clf()
        plt.semilogy(alllosses)
        plt.xlabel('Iterations (1 sample per iteration)')
        plt.ylabel('Loss')
        plt.savefig(dest)
        plt.close()
        # Load the trained model
        model = torch.load(os.path.join(inpath,modelfile))
        return test_idx,train_idx,model


def loss_vs_sense(resultfile='dev_result.pkl',\
    train_log='train_logfile.txt',folder_prefix='run_',outfile='error_analysis.png'):
    '''
    This function reads the model evaluation results (obtained by executing 
    evaluate_model function) and plots the losses with respect to model
    parameters (sense_dim).
    The results are assumed to be located in different folders starting
    with the same prefix in the ted_data_path.
    The output plot is saved in outfile. However, if outfile is None, no
    plot is saved.
    '''
    infolders = glob.glob(os.path.join(ted_data_path,folder_prefix)+'*')
    senselist=[]
    trainlosslist=[]
    testlosslist=[]
    combined = {}
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
        for akey in results:
            if results[akey] and akey!='order':
                if not akey in combined:
                    combined[akey] = [results[akey]]
                else:
                    combined[akey].append(results[akey])
    # Sort results
    idx = np.argsort(senselist)
    senselist,trainlosslist = zip(*[(senselist[i],trainlosslist[i]) \
        for i in idx])
    for akey in combined:
        combined[akey] = [combined[akey][i] for i in idx]

    # Plot the numbers
    if outfile:
        # Plot the loss
        name,ext = ''.join(outfile.split('.')[:-1]),'.'+outfile.split('.')[-1]
        outfilename = os.path.join(ted_data_path,name+'_loss'+ext)
        plt.figure(1)
        plt.clf()
        plt.plot(senselist,trainlosslist,color='blue',label='Train Loss')
        plt.plot(senselist,combined['average_loss'],\
            color='red',label='Test Loss')
        plt.xlabel('Sense Vector Length (Model Complexity)')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(outfilename)
        plt.close()
        print 'Loss figure saved in:',outfilename

        for akey in combined:
            if akey=='average_loss':
                continue
            # Plot other results
            outfilename = os.path.join(ted_data_path,name+'_'+akey+ext)
            plt.figure(2)
            plt.clf()
            
            plt.plot(senselist,combined[akey])
            plt.xticks(senselist)
            plt.grid(True)
            plt.xlabel('Sense Vector Length (Model Complexity)')
            plt.ylabel('Metric Value')
            plt.legend(results['order'])
            plt.title('Test Result for '+akey)
            plt.savefig(outfilename)
            plt.close()
            print akey+' figure saved in:',outfilename

    return senselist, trainlosslist, testlosslist,combined

def average_results(result_pklfilename='dev_result.pkl',folder_prefix='run_'):
    '''
    This function reads the pickle files containing model evaluation results
    (obtained by executing evaluate_model function) and computes the average.
    These pickle files are assumed to be located in different folders starting
    with the same prefix in the ted_data_path. 
    '''
    infolders = glob.glob(os.path.join(ted_data_path,folder_prefix)+'*')
    combined={}
    for afolder in infolders:
        results = cp.load(open(os.path.join(afolder,result_pklfilename)))
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

def tabulate_classical_results(outfile='result_classical.csv'):
    '''
    Read and tabulate result files for classical experiments
    '''
    rating_names = ['beautiful',
                    'ingenious',
                    'fascinating',
                    'obnoxious',
                    'confusing',
                    'funny',
                    'inspiring',
                    'courageous',
                    'ok',
                    'persuasive',
                    'longwinded',
                    'informative',
                    'jaw-dropping',
                    'unconvincing']
    rating_order = ['Prec','Recall','fscore','Accuracy','AUC']
    result_attributes = ['classifier_type',
                         'c_scale',
                         'modalities_used',
                         'lowerthresh_Y',
                         'upperthresh_Y',
                         'scale_rating']
    summary_results =   ['max_results','avg_results',]
    other =['best_classifier','data_normalizer']
      
    outfilename = os.path.join(ted_data_path,'TED_stats/'+outfile)
    infilenames = glob.glob(os.path.join(ted_data_path,'TED_stats/results_*'))
    with open(outfilename,'wb') as fout:
        header = []
        for i,afile in enumerate(infilenames):
            data = cp.load(open(afile))
            # First time
            if i == 0:
                headers = [att for att in result_attributes]
                headers += [akey for akey in data['avg_results']]
                headers += [akey for akey in data['max_results']]
                headers += [arating+'_'+lab for arating in rating_names \
                    for lab in rating_order]
                writer = csv.DictWriter(fout,headers)
                writer.writeheader()
            # Create rows
            arow = {att:data[att] for att in result_attributes \
                if not att=='modalities_used'}
            arow.update({'modalities_used':'_'.join(data['modalities_used'])})
            arow.update(data['avg_results'])
            arow.update(data['max_results'])
            arow.update({arating+'_'+lab:data[arating][i] for arating in rating_names \
                    for i,lab in enumerate(rating_order)})
            writer.writerow(arow)
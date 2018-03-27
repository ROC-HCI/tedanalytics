import os
import json
import glob
import csv
import cPickle as cp
import numpy as np
import scipy as sp

import ted_talk_makeindex
import list_of_talks as lst_talks
import ted_talk_sentiment as ts
import ted_talk_lexical_feature_processor as lex
from TED_data_location import ted_data_path, wordvec_path

import torch
import torch.autograd as autograd

def split_train_test(train_ratio=0.8,talklist=lst_talks.all_valid_talks):
    '''
    Split the list of valid talk id's in training and test sets.
    '''
    m = len(talklist)
    train = np.random.choice(talklist,int(train_ratio*float(m)),\
        replace=False).tolist()
    test = list(set(talklist) - set(train))
    return train,test

def generate_dep_tag(talk_id,dep_type='recur',tag_type='{LG}'):
    '''
    Given a talk_id, it generates the dependency tree and the tag 
    information. The dep_type can take two values: 'recur' means 
    dependency tree in recursive format, 'conll' means conll format.
    The tag_type takes two values: '{LG}' means laughter followed by the
    sentence. {NS} means applause
    '''
    filename = os.path.join(ted_data_path,'TED_meta/'+str(talk_id)+'.pkl')
    data = cp.load(open(filename))
    for (i,j),adep in zip(data['dep_2_fave'],\
            data['dep_trees_'+dep_type]):
        # Tag label
        label = 1 if \
                data['fave_style_transcript'][i]['labels'][j]==tag_type\
                else 0
        # sentence
        sent = data['fave_style_transcript'][i]['sentences'][j]
        # yield only one kind of dependency
        yield adep,label,(i,j),sent

def get_dep_rating(talk_id,scale_rating=True):
    ''' 
    Given a talk_id, it generates the dependency tree and rating.
    '''
    filename = os.path.join(ted_data_path,'TED_meta/'+str(talk_id)+'.pkl')
    data = cp.load(open(filename))
    alldat = []
    ratedict = data['talk_meta']['ratings']
    # Processing the ratings
    if scale_rating:
        ratedict_processed = {akey:float(ratedict[akey])/float(\
            ratedict['total_count']) for akey in sorted(ratedict)\
            if not akey=='total_count'}
    else:
        ratedict_processed = {akey:float(ratedict[akey]) for akey \
            in sorted(ratedict) if not akey=='total_count'}
    # Another convenient format
    y = [vals for keys,vals in sorted(ratedict_processed.items())]
    # All the dependency trees
    for adep in data['dep_trees_recur']:
        alldat.append(adep)
    return alldat,y,ratedict_processed

def binarized_ratings(firstThresh=50.,secondThresh=50.,scale_rating=True):
    '''
    Divides the ground truth ratings into two classes for classification.
    Returns the binarized labels, the median values, and the label of each
    column of y.
    '''
    y_gt = []
    # Read ratings from all the talks
    labels = [label for label in lst_talks.rating_labels \
        if not label=='total_count']
    for atalk in lst_talks.all_valid_talks:
        y=[]
        for akey in lst_talks.rating_labels:
            if not akey=='total_count' and scale_rating:
                y.append(float(lst_talks.all_ratings[atalk][akey])/float(\
                    lst_talks.all_ratings[atalk]['total_count']))
            elif not akey=='total_count' and not scale_rating:
                y.append(float(lst_talks.all_ratings[atalk][akey]))
        y_gt.append(y)
    # Convert into a numpy array
    y_gt = np.array(y_gt)
    # Calculate the median
    thresh1 = sp.percentile(y_gt,firstThresh,axis=0)
    thresh2 = sp.percentile(y_gt,firstThresh,axis=0)
    # Binarize in matrix format for speed
    for i in range(np.size(y_gt,axis=1)):
        y_gt[y_gt[:,i]<=thresh1[i],i] = -1
        y_gt[(thresh1[i]<y_gt[:,i])*(y_gt[:,i]<thresh2[i]),i] = 0
        y_gt[y_gt[:,i]>thresh2[i],i] = 1

    # Convert to dictionary for convenience
    if firstThresh == secondThresh:
        return {key:val.tolist() for key,val in \
            zip(lst_talks.all_valid_talks,y_gt)},thresh1,labels
    else:
        return {key:val.tolist() for key,val in \
            zip(lst_talks.all_valid_talks,y_gt)},thresh1,labels,thresh2

def read_openpose_feat(csv_name =
    'TED_feature_openpose/summary_presenter_openpose_features_0.65.csv'):
    '''
    Read the openpose features
    '''
    filename = os.path.join(ted_data_path,csv_name)
    csvreader = csv.DictReader(open(filename,'rU'))
    X={}
    labels = None
    for arow in csvreader:
        # Read the labels for the first time
        if not labels:
            labels = sorted([label for label in arow if not \
                (label=='root' or label.endswith('_1'))])
        talkid = int(arow['root'])
        if talkid in lst_talks.all_valid_talks:
            X[talkid] = [float(arow[label]) if arow[label]!='nan' else 0. \
                for label in labels]
    return X, labels

def read_openface_feat(csv_name =
    'TED_feature_openface/summary_presenter_openface_features_0.65_header.csv'):
    '''
    Read the openface features
    '''
    filename = os.path.join(ted_data_path,csv_name)
    csvreader = csv.DictReader(open(filename,'rU'))
    X={}
    labels = None
    for arow in csvreader:
        # Read the labels for the first time
        if not labels:
            labels = sorted([label for label in arow if \
                (label!='root' and \
                    ('gaze' in label or \
                        ('AU' in label and \
                         label.endswith('_r'))))])
        talkid = int(arow['root'])
        if talkid in lst_talks.all_valid_talks:
            X[talkid] = [float(arow[label]) if arow[label]!='nan' else 0. \
                for label in labels]
    return X,labels

def read_prosody_feat(talklist=lst_talks.all_valid_talks,
    foldername = 'TED_feature_prosody/full_video'):
    '''
    Reads the prosody features for each video
    '''
    pathname = os.path.join(ted_data_path,foldername)
    X={}
    labels = None
    for atalk in talklist:
        pklname = os.path.join(pathname,str(atalk)+'.pkl')
        if not os.path.exists(pklname):
            print 'Not found',atalk
            continue
        data = cp.load(open(pklname))
        if not labels:
            labels = ['intensity_'+akey for akey in data['intensity'].keys()]+\
                ['pitch_'+akey for akey in data['pitch'].keys()]+\
                ['intensity_'+akey+'_'+str(i) for akey in \
                    data['intensity'].keys() for i in range(5)]
        X[atalk] = data['intensity'].values()+\
            data['pitch'].values()+\
            np.concatenate(data['formant'].values(),axis=0).tolist()

    return X, labels

def read_lexical_feat(talklist=lst_talks.all_valid_talks,
    feature_file = 'misc/lexical.csv'):
    '''
    Reads the lexical features
    '''
    X = {}
    talkset = set(talklist)
    featurefile = os.path.join(ted_data_path,feature_file)
    reader = csv.DictReader(open(featurefile))
    labels = reader.fieldnames[1:]
    for arow in reader:
        atalk = int(arow['TalkID'])
        if not atalk in talkset:
            continue
        X[atalk] = [arow[alabel] for alabel in labels]
    return X, labels

def read_sentiment_feat(talklist=lst_talks.all_valid_talks,
    cacheFolder='misc/'):
    '''
    Read the summary statistics for narrative trajectory features.
    Reading the raw sentiment scores and processing it takes a lot
    of time. Thefore, this function tries to cache the results on
    its first call and save the features in the cache folder.
    '''
    cachepath = os.path.join(ted_data_path,cacheFolder)
    cachefile = os.path.join(cachepath,'processed_sentiment_scores.pkl')
    if os.path.exists(cachefile):
        X,labels = cp.load(open(cachefile))
        if set(X.keys())==set(talklist):
            return X,labels
    comp = ts.Sentiment_Comparator({'all_talks':talklist})
    scorelist = comp.sentiments_interp.values()
    talklist = comp.sentiments_interp.keys()
    X,labels = __compute_summary__(scorelist,comp.column_names)
    # Convert to dictionary
    X = {akey:vals for akey,vals in zip(talklist,X)}
    cp.dump((X,labels),open(cachefile,'wb'))
    return X,labels

def __compute_summary__(scores,col_names):
    '''
    Calculate the summary statistics of the scores such as min, max, 
    average, standard deviation etc.
    Dimension of scores: 
    Number of Talks x interp Talk length (100) x score type (13)
    '''
    X = np.min(scores,axis=1)
    labels = [col+'_min' for col in col_names]
    # Concat maxmimum of the scores
    X = np.concatenate((X,np.max(scores,axis=1)),axis=1)
    labels+= [col+'_max' for col in col_names]
    # Concat average of the scores 
    X = np.concatenate((X,np.mean(scores,axis=1)),axis=1)
    labels+= [col+'_avg' for col in col_names]
    # Concat standard deviation of the scores 
    X = np.concatenate((X,np.std(scores,axis=1)),axis=1)
    labels+= [col+'_std' for col in col_names]
    return X.tolist(), labels

def concat_features(X1, label1, X2, label2):
    '''
    Concatenate two feature dictionaries and corresponding labels
    '''
    labels = label1+label2
    commontalks = list(set(X1.keys()).intersection(set(X2.keys())))
    X = {}
    for atalk in commontalks:
        X[atalk] = X1[atalk]+X2[atalk]
    return X,labels

def read_dep_pos_vocab():
    '''
    Returns list and index dictionary for dependency types and POS's.
    returns a tuple of dependency list, dependency index dictionary,
    parts of speech list, and parts of speech index dictionary.
    '''
    filename = os.path.join(ted_data_path,'misc/dep_pos_list.pkl')
    data = cp.load(open(filename))
    return data['deplist'],data['depidx'],data['poslist'],data['posidx']

def build_ted_vocab():
    '''
    Constructs a vocabulary for TED dataset
    '''
    ted_vocab=set()
    ted_meta_path = os.path.join(ted_data_path,'TED_meta/*.pkl')
    # Create a folder to move the meta files that couldnot be processed
    ted_meta_unprocessed_path = os.path.join(ted_data_path,\
        'TED_meta_unprocessed/')
    if not os.path.exists(ted_meta_unprocessed_path):
        os.makedirs(ted_meta_unprocessed_path)
    # Process the meta files
    for atalk in glob.glob(ted_meta_path):
        path,filename = os.path.split(atalk)
        print filename
        try:
            # Travarse through all dependency trees for all words
            for adep,_,_,_ in generate_dep_tag(int(filename[:-4])):
                allwords = __traverse__(adep)
                for aword in allwords:
                    # Add the word
                    ted_vocab.add(aword)
                    # If the word contains hyphen or period, split them and
                    # add them as well. Also add a hypen and periodless 
                    # version because these things often cause confusions. 
                    # This will be required when the engine would try to 
                    # resolve unfound words.
                    if '-' in aword:
                        ted_vocab.update(aword.split('-'))
                        ted_vocab.add(aword.replace('-',''))
                    if '.' in aword:
                        ted_vocab.update(aword.split('.'))
                        ted_vocab.add(aword.replace('.',''))
        except Exception as e:
            print e
            print 'Moving the meta file {} to {}'.format(atalk,\
                ted_meta_unprocessed_path)
            os.rename(atalk,os.path.join(ted_meta_unprocessed_path,filename))
    print 'Total size of vocabulary:',len(ted_vocab)
    return ted_vocab

def crop_glove_dictionary():
    '''
    This function creates a reduced version of the glove word2vec dictionary 
    by deleting all the words that did not appear in the ted talk vocabulary.
    '''
    print 'building ted vocab'
    ted_vocab = build_ted_vocab()
    with open(wordvec_path) as fin:
        with open(wordvec_path+'_cropped','wb') as fout:
            rows=[]
            print 'reading glove w2v'
            for i,aline in enumerate(fin):
                if i%100000==0:
                    print i
                splt_line = aline.strip().split()
                # Check if the word is found in ted vocabulary 
                if splt_line[0] in ted_vocab:
                    rows.append(splt_line)
                    ted_vocab.remove(splt_line[0])
            # Writing the cropped glove dictionary
            for arow in sorted(rows):
                fout.write(' '.join(arow)+'\n')

def read_crop_glove():
    '''
    Reads the cropped glove file and returns a dictionary mapping every
    word into its corresponding glove vector.
    '''
    retdict = {}
    with open(wordvec_path+'_cropped') as fin:
        for aline in fin:
            splt_line = aline.strip().split()
            val = map(float,splt_line[1:])
            retdict[splt_line[0]] = val
    return retdict


def __traverse__(atree):
    '''
    Recursively travarses and returns the words in a dependency tree
    '''
    words=set()
    for anode in atree:
        if type(anode)==str or type(anode)==unicode:
            w,p,d = anode.strip().encode('ascii','ignore').split()
            words.add(w)
        elif type(anode)==list:
            words.update(__traverse__(anode))
    return words

def __tree_rating_feeder__(atalk,gpunum=-1):
    '''
    Helper function to feed the dependency trees and the ratings to the model.
    It transforms the data and the ground truth in appropriate format so that
    it can be used as a feeder-function argument for train_model function.
    '''
    # All the dependency trees and the rating
    all_deptree,y,_ = get_dep_rating(atalk)
    # Construct ground truth tensor
    if gpunum < 0:
        rating_t = autograd.Variable(torch.FloatTensor(y))
    else:
        with torch.cuda.device(gpunum):
            rating_t = autograd.Variable(torch.cuda.FloatTensor(y))
    return all_deptree,rating_t.view(1,-1)

def main():
    '''
    This function prepares the dataset for the code. It must run at least
    once after linking the code with the dataset. Before running, please
    make sure that the TED_data_location.py file is updated with the correct
    data location.
    '''
    print 'preparing data'
    print 'please make sure that the TED_data_location.py file is updated'+\
        'with the correct data location'
    print 'making index'
    ted_talk_makeindex.main()
    # Reloading the talk list
    reload(lst_talks)
    print 'Preparing the glove dictionary'
    crop_glove_dictionary()
    print 'Preparing Sentiment features'
    read_sentiment_feat()
    print 'Preparing Storytelling features'
    import ted_talk_classical_experiments as ttce
    X,_,_,comp = ttce.__loaddata__()
    ttce.evaluate_clusters_pretty()
    print 'Preparing Lexical Features'
    lex.prepare_lexical_feat()

if __name__=='__main__':
    main()


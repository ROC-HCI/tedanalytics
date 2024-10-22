import os
import re
import json
import glob
import csv
import threading
import cPickle as cp
import numpy as np
import scipy as sp
from functools import partial
import multiprocessing as mp
from itertools import izip_longest
from nltk.tokenize import word_tokenize

import ted_talk_makeindex
import list_of_talks as lst_talks
import ted_talk_sentiment as ts
import ted_talk_lexical_feature_processor as lex
from TED_data_location import ted_data_path, wordvec_path

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

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

def get_dep_rating(talk_id,scale_rating=True,process_ratings=True):
    ''' 
    Given a talk_id, it generates the dependency tree and rating.
    '''
    filename = os.path.join(ted_data_path,'TED_meta/'+str(talk_id)+'.pkl')
    data = cp.load(open(filename))
    alldat = []
    # Provision to skip the ratings altogether
    if process_ratings:
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
    if process_ratings:
        return alldat,y,ratedict_processed
    else:
        return alldat

def rating_ratio():
    binrats,labels = binarized_ratings()
    count=[]
    for i,avalidtalk in enumerate(binrats):
        if i==0:
            count = [1 if x==1. else 0 for x in binrats[avalidtalk]]
        count = [x+1 if y==1. else x for x,y in zip(count,binrats[avalidtalk])]
        print count
    count = [arat/float(len(binrats)) for arat in count]
    return {l:val for l,val in zip(labels,count)}


def binarized_ratings(firstThresh=50.,secondThresh=50.,scale_rating=True,
    read_hidden_test_set=False):
    '''
    Divides the ground truth ratings into two classes for classification.
    Returns the binarized labels, the median values, and the label of each
    column of y.
    '''
    valid_talks = lst_talks.all_valid_talks
    valid_ratings = lst_talks.all_ratings.copy()
    if read_hidden_test_set:
        # When the reserved Test dataset is accessed, inform the user about it
        # as a precausion.
        print 'Warning: Accessing the hidden test set'
        valid_talks = lst_talks.all_valid_talks + lst_talks.test_set
        valid_ratings.update(lst_talks.test_set_ratings)
    y_gt = []
    # Read ratings from all the talks
    labels = [label for label in lst_talks.rating_labels \
        if not label=='total_count']
    for atalk in valid_talks:
        y=[]
        for akey in lst_talks.rating_labels:
            if not akey=='total_count' and scale_rating:
                y.append(float(valid_ratings[atalk][akey])/float(\
                    valid_ratings[atalk]['total_count']))
            elif not akey=='total_count' and not scale_rating:
                y.append(float(valid_ratings[atalk][akey]))
        y_gt.append(y)
    # Convert into a numpy array
    y_gt = np.array(y_gt)    
    # Calculate the thresholds
    thresh1 = sp.percentile(y_gt,firstThresh,axis=0)
    thresh2 = sp.percentile(y_gt,secondThresh,axis=0)

    # Binarize in matrix format for speed
    cols = np.size(y_gt,axis=1)
    for i in range(cols):
        y_gt[y_gt[:,i]<=thresh1[i],i] = -1
        y_gt[(thresh1[i]<y_gt[:,i])*(y_gt[:,i]<thresh2[i]),i] = 0
        y_gt[y_gt[:,i]>thresh2[i],i] = 1
    
    # Convert to dictionary for convenience
    return {key:val.tolist() for key,val in zip(valid_talks,y_gt)},labels

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
    foldername = 'TED_feature_prosody/full_video_summary'):
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
                ['formant_'+akey+'_'+str(i) for akey in \
                    data['formant'].keys() for i in range(5)]
        X[atalk] = data['intensity'].values()+\
            data['pitch'].values()+\
            [data['formant'][akey][i] for akey in \
                    data['formant'].keys() for i in range(5)]
    return X, labels

def read_prosody_per_sentence(atalk,
    foldername = 'TED_feature_prosody/per_sentence'):
    '''
    Reads the prosody features for each sentence in a given talk id
    '''
    pathname = os.path.join(ted_data_path,foldername)
    pklname = os.path.join(pathname,str(atalk)+'.pkl')
    X=[]    
    if not os.path.exists(pklname):
        raise IOError('Not found'+str(atalk))
    labels = None
    for data in cp.load(open(pklname))['sentences']:
        if not data['intensity'] or not data['formant'] or not data['pitch']:
            X.append([])
            continue
        if not labels:
            labels = ['intensity_'+akey for akey in data['intensity'].keys()]+\
                ['pitch_'+akey for akey in data['pitch'].keys()]+\
                ['formant_'+akey+'_'+str(i) for akey in \
                    data['formant'].keys() for i in range(5)]
        asent = data['intensity'].values()+\
            data['pitch'].values()+\
            [data['formant'][akey][i] for akey in \
                    data['formant'].keys() for i in range(5)]
        X.append(asent)
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
    # Concat standard deviation of the scores 
    X = np.concatenate((X,sp.stats.skew(scores,axis=1)),axis=1)
    labels+= [col+'_skew' for col in col_names]
    # Concat standard deviation of the scores 
    X = np.concatenate((X,sp.stats.kurtosis(scores,axis=1)),axis=1)
    labels+= [col+'_kurt' for col in col_names]        
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
        rating_t = Variable(torch.FloatTensor(y))
    else:
        with torch.cuda.device(gpunum):
            rating_t = Variable(torch.cuda.FloatTensor(y))
    return all_deptree,rating_t.view(1,-1)

def gputize(input,gpuNum):
    '''
    Put torch elements to GPU if necessary
    '''
    if gpuNum>=0:
        return input.cuda(gpuNum)
    else:
        return input

def variablize(input,gpuNum):
    '''
    Turn numpy to variable. put to gpu if necessary
    '''
    if gpuNum>=0:
        return Variable(torch.from_numpy(input).cuda(gpuNum))
    else:
        return Variable(torch.from_numpy(input))

def get_minibatch_iter(dataset,minibatch_size,skipkeys={'ylabel'}):
    '''
    Iterator for a shuffled and variablized minibatch. This method correctly
    handles the dataset.flatten_sentence flag.
    '''
    m = len(dataset)
    minibatch_size = min(minibatch_size,m)
    idx = np.arange(m)
    np.random.shuffle(idx)
    if dataset.flatten_sentence:
        for i in range(0,len(idx),minibatch_size):
            batch = []
            for j in range(i,min(len(idx),i+minibatch_size)):
                adata = {akey:dataset[idx[j]][akey] \
                    for akey in dataset[idx[j]] if not akey in skipkeys}
                batch.append(adata)
            yield batch
    else:
        for i in range(0,len(idx),minibatch_size):
            batch = []
            for j in range(i,min(len(idx),i+minibatch_size)):
                adata={}
                for akey in dataset[idx[j]]:
                    if akey in skipkeys:
                        continue
                    # if akey=='X':
                    #     adata[akey]=[asent for asent in dataset[idx[j]][akey]]
                    # else:
                    adata[akey]=dataset[idx[j]][akey]
                batch.append(adata)
            yield batch

def f_map(minb_idx,dataset,datakeys,gpuNum):
    retlist=[]
    for i in minb_idx:
        retlist.append({akey:variablize(dataset[i][akey],gpuNum)\
                for akey in datakeys})
    return retlist

def f_map_pooled(minb_idx,dataset,datakeys,gpuNum,q):
    retlist=[]
    for i in minb_idx:
        retlist.append({akey:variablize(dataset[i][akey],gpuNum)\
            for akey in datakeys})
    q.put(retlist)

def get_minibatch_iter_pooled(dataset,minibatch_size,gpuNum,
    datakeys=['X','Y']):
    '''
    Iterator for a shuffled and variablized minibatch.
    '''
    m = len(dataset)
    minibatch_size = min(minibatch_size,m)
    idx = np.arange(m)
    np.random.shuffle(idx)

    # create a worker pool and the mapping function
    q = mp.Queue()
    processes = mp.cpu_count()-1

    # Create minibatch indices and map using the workers
    #minb_indices = [[j for j in range(i,min(len(idx),i+minibatch_size))] \
    #    for i in range(0,len(idx),minibatch_size)]
    minb_indices = [[j for j in range(i,min(len(idx),i+m/processes))] \
        for i in range(0,len(idx),m/processes)]

    # Run individual processes
    proclist = []
    for i in range(processes):
        p = mp.Process(target=f_map_pooled,\
            args=(minb_indices[i],dataset,datakeys,gpuNum,q,))
        p.daemon = True
        p.start()
        proclist.append(p)

    # Collect all results into a single result list.
    datasetlist = []
    for p in proclist:
        datasetlist.extend(q.get())
      
    # Hold until all done
    for p in proclist:
        p.join()

    return datasetlist

###################### Deprecated #############################
# class TED_Rating_Averaged_Dataset(Dataset):
#     """
#     TED rating dataset. The pose and face values are averaged.
#     Note: Input Semantics for LSTM
#     The first axis is the sequence itself (T), the second indexes
#     instances in the mini-batch (B), and the third indexes elements
#     of the input (*).
#     TODO: 
#     1. Utilize wvec_index_maker
#     2. Wordwise prosody
#     3. Sentence-wise Face and Pose
#     4. Preload into the GPU
#     """

#     def __init__(self, data_indices=lst_talks.all_valid_talks,firstThresh=50.,\
#             secondThresh=50.,scale_rating=True,posedim=36,prosodydim=49,\
#             facedim=44,worddim=300,modality=['word','pose','face','audio'],\
#             flatten_sentence=False,access_hidden_test=False,gpuNum=-1):
#         self.gpunum = gpuNum
#         # get ratings
#         self.Y,self.ylabel = binarized_ratings(firstThresh,\
#             secondThresh,scale_rating)
#         # Indices of the data in the dataset
#         self.data_indices = list(set(data_indices).intersection(self.Y.keys()))
#         ################ DEBUG * REMOVE ###############
#         #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         #self.data_indices = self.data_indices[:10]
#         #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
#         ###############################################
        
#         # Other important information for the dataset
#         print 'reading word vectors'
#         # Reading the complete w2v dictionary
#         self.wvec = read_crop_glove()
#         print 'reading ratings'
#         self.ratings = {akey:self.Y[akey] for akey in self.data_indices}
#         self.posepath = os.path.join(ted_data_path,\
#             'TED_feature_openpose/openpose_sentencewise_features/')
#         self.facepath = os.path.join(ted_data_path,\
#             'TED_feature_openface/openface_sentencewise_features/')
#         self.sentpath = os.path.join(ted_data_path,\
#             'TED_feature_sentence_boundary/')
#         self.prospath = os.path.join(ted_data_path,\
#             'TED_feature_prosody/per_sentence')
#         self.modality = modality
#         self.posedim = posedim
#         self.prosdim = prosodydim
#         self.facedim = facedim
#         self.worddim = worddim

#         self.dims = 0
#         if 'pose' in self.modality:
#             self.dims += posedim
#         if 'face' in self.modality:
#             self.dims += facedim
#         if 'word' in self.modality:
#             self.dims += worddim
#         if 'audio' in self.modality:
#             self.dims += prosodydim

#         # Final check for the existence of all the files.
#         all_available = set()
#         print 'Checking indices'
#         for atalk in self.data_indices:
#             posefile = os.path.join(self.posepath,str(atalk)+'.pkl')
#             facefile = os.path.join(self.facepath,str(atalk)+'.pkl')
#             sentfile = os.path.join(self.sentpath,str(atalk)+'.pkl')
#             prosfile = os.path.join(self.prospath,str(atalk)+'.pkl')
#             if os.path.exists(posefile) and os.path.exists(facefile) \
#                 and os.path.exists(sentfile) and os.path.exists(prosfile):
#                 all_available.add(atalk)
#         # Remove indices that are not all available
#         self.data_indices = list(all_available.intersection(self.data_indices))
#         print 'Dataset Ready'

        
#     def __len__(self):
#         return len(self.data_indices)

#     def __getitem__(self, idx):
#         '''
#         Given a data index (not videoid), get the corresponding data
#         '''
#         talkid = self.data_indices[idx]
#         posedat = cp.load(open(os.path.join(self.posepath,str(talkid)+'.pkl')))
#         facedat = cp.load(open(os.path.join(self.facepath,str(talkid)+'.pkl')))
#         facedat = facedat[0]
#         sentdat = cp.load(open(os.path.join(self.sentpath,str(talkid)+'.pkl')))
#         sentdat = sentdat['sentences']
#         prosdat,_ = read_prosody_per_sentence(talkid)

#         outvec=[]
#         # Second best candidate for data parallelization. CPU multithreading
#         for asent,aface,apose,apros in izip_longest(sentdat,facedat,\
#                 posedat,prosdat,fillvalue=[]):
#             # Average pose per sentence
#             if 'pose' in self.modality:
#                 if not apose:
#                     apose = np.zeros(self.posedim)
#                 else:
#                     assert self.posedim == np.size(apose,axis=1)
#                     apose = np.nanmean(np.nan_to_num(apose),axis=0)
#             # Average face per sentence
#             if 'face' in self.modality:
#                 if not aface:
#                     aface = np.zeros(self.facedim)
#                 else:
#                     assert self.facedim == np.size(aface,axis=1)
#                     aface = np.nanmean(np.nan_to_num(aface),axis=0)
#             # Average, std, skew, kartosys of prosody per sentence
#             if 'audio' in self.modality:
#                 if not apros:
#                     apros = np.zeros(self.prosdim)
#                 else:
#                     apros = np.nan_to_num(apros)

#             # Concatenate pose and face and prosody
#             if 'pose' in self.modality and 'face' in self.modality:
#                 nonverbal = np.concatenate((apose,aface,apros)).tolist()
#             elif 'pose' in self.modality:
#                 nonverbal = apose
#             elif 'face' in self.modality:
#                 nonverbal = aface
#             elif 'audio' in self.modality:
#                 nonverbal = apros
#             else:
#                 nonverbal = []

#             # Take all words in a sentence
#             if 'word' in self.modality:
#                 allwords=[]
#                 nullvec = [0 for i in range(self.worddim)]
#                 for aword in word_tokenize(asent['sentence']):
#                     if aword in self.wvec:
#                         allwords.append(self.wvec[aword]+nonverbal)
#                     else:
#                         allwords.append(nullvec+nonverbal)
#                 outvec.extend(allwords)
#             else:
#                 outvec.extend([nonverbal])
#         # Note: This is T x * (not T x B (batch = 1) x *
#         # Because the collate function needs in this form for
#         # pad_sequence
#         xval = np.array(outvec).astype(np.float32)

#         yval = np.reshape([1 if alab == 1 else 0 for alab in \
#             self.Y[talkid]],(1,-1)).astype(np.int64)
#         return {'X':xval,'Y':yval,'ylabel':self.ylabel}

def collate_and_pack_simple(datalist,gpuNum):
    '''
    Pad and pack a list of datapoints obtained from 
    TED_Rating_Averaged_Dataset
    or
    TED_Rating_wordonly_indices_Dataset
    '''
    # Packing X 
    sizelist = [np.size(item['X'],axis=0) for item in datalist]
    idx = np.argsort([-asize for asize in sizelist])
    sizelist = [sizelist[i] for i in idx]
    X_batch = []
    Y_batch = []
    # Sorting based on size
    for i in idx:
        X_batch.append(variablize(datalist[i]['X'],gpuNum))
        Y_batch.append(datalist[i]['Y'])
    # Padded the data dimension is T x B (batchsize) x *
    padded = pad_sequence(X_batch)
    # Datapack might give a wrong impression of dimension but its ok
    datapack = pack_padded_sequence(padded,sizelist)
    # Arranging Y
    Y_shaped = np.concatenate(Y_batch,axis=0)
    Y = variablize(Y_shaped,gpuNum)
    return {'X':datapack,'Y':Y}

class TED_Rating_wordonly_indices_Dataset(Dataset):
    """
    TED rating dataset.
    Only the word modality is used. It will pass the indices of the
    word-vectors for each word in the transcript. For out of vocabulary
    word, it will put -1 as the index. For compound words (e.g. modern-world),
    it will juxtapose the indices of the constituent words.
    The data is gputized and variablized.

    Note: Input Semantics for LSTM
    The first axis is the sequence itself (T), the second indexes
    instances in the mini-batch (B), and the third indexes elements
    of the input (*).
    """

    def __init__(self, data_indices=lst_talks.all_valid_talks,firstThresh=50.,\
            secondThresh=50.,scale_rating=True,flatten_sentence=False,
            access_hidden_test=False,gpuNum=-1):

        self.gpunum = gpuNum
        # get ratings
        self.Y,self.ylabel = binarized_ratings(firstThresh,\
            secondThresh,scale_rating,read_hidden_test_set=access_hidden_test)
        # Indices of the data in the dataset
        self.data_indices = list(set(data_indices).intersection(self.Y.keys()))
        ################ DEBUG * REMOVE ###############
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # self.data_indices = self.data_indices[:2]
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        ###############################################
        print 'reading ratings'
        self.ratings = {akey:self.Y[akey] for akey in self.data_indices}

        # Other important information for the dataset
        print 'reading word vectors'
        self.flatten_sentence=flatten_sentence
        self.wvec_map = wvec_index_maker(self.gpunum)
        self.dims = self.wvec_map.dims
        print 'Dataset Ready'

        
    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        '''
        Given a data index (not videoid), get the corresponding data
        '''
        talkid = self.data_indices[idx]
        outvec = self.wvec_map(talkid,self.flatten_sentence)        
        # Note: This is T x * (not T x B (batch = 1) x *
        # Because the collate function needs in this form for
        # pad_sequence
        # Note 2: The inputs have only one dimension (w2v index) in
        # this case, i.e. length along * dimension is 1
        # Note 3: the x values are converted to numpy equivalent of
        # tensor.Longtensor because these are indices and will be supplied
        # as index intigers in pytorch later on.
        if self.flatten_sentence:
            xval = outvec.reshape(-1,1)
        else:
            xval=[asent.reshape(-1,1) for asent in outvec]
        # Process the labels
        yval = variablize(np.reshape([1 if alab == 1 else 0 for alab in \
            self.Y[talkid]],(-1,1)).astype(np.float32),self.gpunum)
        return {'X':xval,'Y':yval,'ylabel':self.ylabel}

class TED_Rating_depPOSonly_indices_Dataset(Dataset):
    """TED rating dataset.
    Only the dependency type and POS are used; arranged within the tree.
    The data is gputized and variablized. This dataset doesn't support
    flatten_sentence = True.
    It must be processed using a recursive neural network.
    """

    def __init__(self, data_indices=lst_talks.all_valid_talks,firstThresh=50.,\
            secondThresh=50.,scale_rating=True,flatten_sentence=False,
            access_hidden_test=False,wvec_index_maker=None,gpuNum=-1):
        self.gpunum = gpuNum        
        # get ratings
        self.Y,self.ylabel = binarized_ratings(firstThresh,\
            secondThresh,scale_rating,read_hidden_test_set=access_hidden_test)
        # Indices of the data in the dataset
        self.data_indices = list(set(data_indices).intersection(self.Y.keys()))
        ################ DEBUG * REMOVE ###############
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #self.data_indices = self.data_indices[:10]
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        ###############################################
        print 'reading ratings'
        self.ratings = {akey:self.Y[akey] for akey in self.data_indices}

        # Other important information for the dataset
        print 'reading vocabs'
        self.flatten_sentence=flatten_sentence
        if flatten_sentence:
            raise IOError('This dataset does not support flattening')
        _,self.depidx,_,self.posidx = read_dep_pos_vocab()
        vardeps = variablize(np.array(\
            self.depidx.values()).astype(np.int64),self.gpunum)
        varposes = variablize(np.array(\
            self.posidx.values()).astype(np.int64),self.gpunum)
        self.depidx = {dep:a_vardep for dep,a_vardep in \
            zip(self.depidx.keys(),vardeps)}
        self.posidx = {pos:a_varpos for pos,a_varpos in \
            zip(self.posidx.keys(),varposes)}
        if wvec_index_maker:
            self.wvec_map = wvec_index_maker
            self.oov = variablize(np.zeros(300).astype(np.float32),self.gpunum)
        else:
            self.wvec_map = None
        # Just the dependency index and pos index
        if not wvec_index_maker:
            self.dims = 2
        else:
            self.dims = 3
        print 'Dataset Ready'

        
    def __len__(self):
        return len(self.data_indices)

    def __convert_atree__(self, atree):
        '''
        Represent a dependency tree with variablized indices
        '''
        n = len(atree)
        for i in range(n):
            if type(atree[i]) in [str, unicode]:
                w,p,d = atree[i].strip().encode('ascii','ignore').split()
                if not self.wvec_map:
                    atree[i] = (self.depidx[d],self.posidx[p])
                else:
                    if w in self.wvec_map.w2v_indices:
                        idx = self.wvec_map.w2v_indices[w]
                        wvec = self.wvec_map.w2v_vals[idx]
                    else:
                        wvec = self.oov
                    atree[i] = (self.depidx[d],self.posidx[p],wvec)
            elif type(atree[i])==list:
                atree[i] = self.__convert_atree__(atree[i])
        return atree


    def __getitem__(self, idx):
        '''
        Given a data index (not videoid), get the corresponding data
        '''
        talkid = self.data_indices[idx]
        deptrees = get_dep_rating(talkid,process_ratings=False)
        coded_dtrees = map(self.__convert_atree__,deptrees)
        # Process the labels
        yval = variablize(np.reshape([1 if alab == 1 else 0 for alab in \
            self.Y[talkid]],(-1,1)).astype(np.float32),self.gpunum)
        return {'X':coded_dtrees,'Y':yval,'ylabel':self.ylabel}

def __read_ooTextFile__(filename):
    '''
    Read the prosody files
    '''
    dir_pros = os.path.join(ted_data_path,'TED_feature_prosody',
        'full_video_raw')
    dat = open(os.path.join(dir_pros,filename)).read().split('\n')
    if not dat:
        raise IOError('Empty file')
    if not dat[0]=='File type = "ooTextFile"':
        raise IOError('File type not recognized')
    if not dat[1] in {'Object class = "Pitch 1"',
        'Object class = "Intensity 2"', 'Object class = "Formant 2"'}:
        raise IOError('Object class not recognized')

    if dat[1].endswith('"Pitch 1"'):        
        samples = []
        filelen = len(dat)
        i = 0
        while i < filelen:
            if dat[i].strip().startswith('nx'):
                N = int(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('dx'):
                rate = 1/float(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('ceiling'):
                ceiling = float(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('nCandidates'):
                nCandidates = int(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('candidate [1]'):
                pitch_freq = float(dat[i+1].split('=')[-1].strip())
                samples.append(pitch_freq)
                i += (nCandidates-1)*3 + 2
                continue
            i+=1
        return {'rate':rate,'N':N,'ceiling':ceiling,\
            'samples':np.array(samples,dtype=np.float32)[None]}
    elif dat[1].endswith('"Intensity 2"'):
        samples = []
        filelen = len(dat)
        i = 0
        while i < filelen:
            if dat[i].strip().startswith('nx'):
                N = int(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('dx'):
                rate = 1/float(dat[i].split('=')[-1].strip())
            else:
                output = re.match(r'z \[1\] \[\d*\] = (-?\d*\.?\d*)',\
                    dat[i].strip())
                if output:
                    loud = float(output.group(1))
                    samples.append(loud)
            i+=1
        return {'rate':rate,'N':N,'samples':np.array(samples,dtype=np.float32)[None]}
    elif dat[1].endswith('"Formant 2"'):
        samples = []
        filelen = len(dat)
        i = 0
        while i < filelen:
            if dat[i].strip().startswith('nx'):
                N = int(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('dx'):
                rate = 1/float(dat[i].split('=')[-1].strip())
            elif dat[i].strip().startswith('maxnFormants'):
                maxnFormants = int(dat[i].split('=')[-1].strip())
                maxnFormants = min(maxnFormants,3)
            elif dat[i].strip().startswith('nFormants'):
                nFormants = int(dat[i].split('=')[-1].strip())
                formantlist = [0.]*(maxnFormants*2)
            elif dat[i].strip().startswith('formant'):
                output = re.match(r'formant \[([1-{0}])\]:'.format(\
                    min(nFormants,maxnFormants)),dat[i].strip())
                if output:
                    i_form = int(output.group(1))
                    form_freq = float(dat[i+1].split('=')[-1].strip())
                    form_bw = float(dat[i+2].split('=')[-1].strip())
                    formantlist[(i_form-1)*2] = form_freq
                    formantlist[(i_form-1)*2+1] = form_bw
                    if i_form == min(nFormants,maxnFormants):
                        samples.append(formantlist)
                    i+=2
                    continue
            i+=1
        return {'rate':rate,'N':N,'samples':np.array(samples,dtype=np.float32).T}

def read_raw_prosody_per_sentence(atalk,num_sentences,gpuNum=-1):
    '''
    Returns interpolated nonverbal features within the
    boundaries of each sentence in the transcript. The signals
    are padded (to the longest sentence) and z-score normalized.
    The mean and std of the signals are also returned.
    '''
    pitch = __read_ooTextFile__(str(atalk)+'.pitch')
    form = __read_ooTextFile__(str(atalk)+'.formant')
    loud = __read_ooTextFile__(str(atalk)+'.loud')
    min_len = min(pitch['N'],form['N'],loud['N'])
    assert pitch['rate']==form['rate']==loud['rate'],'Signal rates are not equal'

    prosody = np.concatenate((pitch['samples'][:,:min_len],\
        loud['samples'][:,:min_len],form['samples'][:,:min_len]),axis=0)
    M,N = prosody.shape

    # Standardize ignoring the zero values
    prosody[prosody==0.]=np.nan
    mean_pros = np.nanmean(prosody,axis=1)[None].T
    std_pros = np.nanstd(prosody,axis=1)[None].T
    prosody = (prosody - mean_pros)/std_pros
    np.nan_to_num(prosody,False)

    sent_bounds = get_sent_boundary(atalk)

    all_chunks = []
    longest = 0
    # Segment the prosody into chunks per sentence
    for sentid in range(num_sentences):
        st = 0
        en = -1
        if sentid in sent_bounds:    
            st = sent_bounds[sentid][0]
            en = sent_bounds[sentid][1]
        l = max(0,int(st*pitch['rate']))
        r = min(int(en*pitch['rate']),N)
        if r<=l:
            # Just take zeros if no prosody corresponding
            # to the sentence is found
            chunk = np.zeros((M,1),dtype=prosody.dtype)
        else:
            chunk = prosody[:,l:r]
            
        longest = max(longest,np.size(chunk,axis=1))
        all_chunks.append(chunk)
    
    # Pad the chunks to the size of the longest one
    for i,a_chunk in enumerate(all_chunks):
        to_pad = longest - np.size(a_chunk,axis=1)
        chunk_padded = np.pad(a_chunk,((0,0),(0,to_pad)),
            'constant',constant_values=0)
        all_chunks[i] = variablize(chunk_padded,gpuNum).view(1,M,-1)
    all_chunks = torch.cat(all_chunks)
    return all_chunks, variablize(mean_pros,gpuNum), \
        variablize(std_pros,gpuNum)

class TED_Rating_depPOSnorverbal_Dataset(TED_Rating_depPOSonly_indices_Dataset):
    """TED rating dataset.
    The dependency type, POS and nonverbal modalities are used.
    The data is gputized and variablized. This dataset doesn't support
    flatten_sentence = True.
    It must be processed using an RNN and a CNN.
    """
    def __getitem__(self, idx):
        '''
        Given a data index (not videoid), get the corresponding data
        '''
        talkid = self.data_indices[idx]
        
        deptrees = get_dep_rating(talkid,process_ratings=False)
        coded_dtrees = map(self.__convert_atree__,deptrees)
        num_sentences = len(coded_dtrees)

        prosody_per_sent,mean_,std_ = read_raw_prosody_per_sentence(talkid,\
            num_sentences,gpuNum=self.gpunum)
        # Process the labels
        yval = variablize(np.reshape([1 if alab == 1 else 0 for alab in \
            self.Y[talkid]],(-1,1)).astype(np.float32),self.gpunum)
        return {'X':coded_dtrees, 'X_pros':prosody_per_sent,
          'X_pros_mean':mean_,'X_pros_std':std_,'Y':yval,
          'ylabel':self.ylabel}

class wvec_index_maker():
    '''
    A class to convert the transcript of any talk to a list of word-vector indices.
    It takes care of the word pre-processing issues. For convenience, it also saves
    the wordvector values (w2v_vals) as a matrix (easier gputization). The rows of
    this matrix is consistent with the w2v_indices. Initialize it only once because
    it loads the w2v dictionary while initializing.
    If load_words is True, it keeps a dictionary that maps from the word-indices to
    the actual words. Loading this dictionary is memory intensive so its usage is
    discouraged.
    '''
    def __init__(self,gpuNum,load_words=False):
        self.gpunum = gpuNum
        # Reading the complete w2v dictionary
        wvec = read_crop_glove()
        self.w2v_indices = {akey:i for i,akey in enumerate(wvec)}
        if load_words:
            self.w2v_words = {i:akey for i,akey in enumerate(wvec)}
        self.w2v_vals = variablize(np.array(wvec.values()).astype(np.float32),gpuNum)
        self.dims = len(wvec['the'])
    
    def __call__(self,talkid,flatten_sentence=False):
        '''
        Read all the sentences from the transcript
        Convert all words to w2v indices using Word2idx function
        '''
        trscpt_loc = os.path.join(ted_data_path,'TED_meta/'+str(talkid)+'.pkl')
        data = cp.load(open(trscpt_loc))
        allidx=[]
        for item in data['fave_style_transcript']:
            for asent in item['sentences']:
                # Filter tags
                temp = asent.replace('{NS}','').replace('{LG}','').strip()
                if temp:
                    wordlist = word_tokenize(temp.lower())
                    if flatten_sentence:
                        allidx.extend(self.word2idx(wordlist).view(-1,1))
                    else:
                        allidx.append(self.word2idx(wordlist))
        if flatten_sentence:
            allidx = torch.cat(allidx,dim = 0)
        return allidx

    def word2idx(self,wordlist):
        # Convert a list of words to list of wordvector indices
        # breaks compound words into constituents
        # puts -1 for out of vocabulary words        
        allidx=[]
        for aword in wordlist:
            if aword in self.w2v_indices:
                allidx.append(self.w2v_indices[aword])
            elif '-' in aword:
                tempwords = aword.split('-')
                allidx.extend(self.word2idx(tempwords))
            elif '.' in aword:
                tempwords = aword.split('.')
                allidx.extend(self.word2idx(tempwords))
            else:
                allidx.append(-1)
        return variablize(np.array(allidx).astype(np.int64),self.gpunum)

    def __display_transcript__(self,talkid,flatten_sentence=False):
        '''
        Given a talkid, shows the words the way it is encoded. This function
        requires loading the w2v_words dictionary by setting load_words arg
        to True requiring additional memory.
        '''
        allidx = self.__call__(talkid,flatten_sentence)
        if flatten_sentence:
            for idx_int in allidx.cpu().numpy().tolist():
                if idx_int == -1:
                    print '<OOV>','\t','-1'
                else:
                    print self.w2v_words[idx_int],'\t',idx_int
        else:
            for senid,asent in enumerate(allidx):
                print 'sentence --',senid
                for anidx in asent.cpu().data.numpy().tolist():
                    if anidx == -1:
                        print '<OOV>','\t','-1'
                    else:
                        print self.w2v_words[anidx],'\t',anidx
                print


########################### Deprecated #########################################
# class TED_Rating_Streamed_Dataset(TED_Rating_Averaged_Dataset):
#     """
#     TED rating dataset. The word, pose and face values are
#     sent as three different streams.
    
#     Note: Input Semantics for LSTM
#     The first axis is the sequence itself (T), the second indexes
#     instances in the mini-batch (B), and the third indexes elements
#     of the input (*).

#     TODO: Utilize wvec_index_maker
#     """
#     def __getitem__(self, idx):
#         '''
#         Given a data index (not videoid), get the corresponding data
#         '''
#         talkid = self.data_indices[idx]
#         posedat = cp.load(open(os.path.join(self.posepath,str(talkid)+'.pkl')))
#         facedat = cp.load(open(os.path.join(self.facepath,str(talkid)+'.pkl')))
#         facedat = facedat[0]
#         sentdat = cp.load(open(os.path.join(self.sentpath,str(talkid)+'.pkl')))
#         sentdat = sentdat['sentences']

#         pose_stream=[]
#         face_stream=[]
#         wrd_stream=[]
#         nullwvec = [0 for i in range(self.worddim)]
#         nullpvec = [[0 for i in range(self.posedim)]]
#         nullfvec = [[0 for i in range(self.facedim)]]
#         retval = {}

#         for asent,aface,apose in izip_longest(sentdat,facedat,posedat,\
#                 fillvalue=[]):
#             # Pose
#             if not apose:
#                 pose_stream.extend(nullpvec)
#             else:
#                 pose_stream.extend(np.nan_to_num(apose).tolist())
#             # Face
#             if not aface:
#                 face_stream.extend(nullfvec)
#             else:
#                 face_stream.extend(np.nan_to_num(aface).tolist())
#             # word
#             allwords=[]
#             for aword in word_tokenize(asent['sentence']):
#                 if aword in self.wvec:
#                     allwords.append(self.wvec[aword])
#                 else:
#                     allwords.append(nullwvec)
#             wrd_stream.extend(allwords)
#         # Selectively add modalities
#         if 'pose' in self.modality:
#             retval['pose'] = np.array(pose_stream).astype(np.float32)
#         if 'face' in self.modality:
#             retval['face'] = np.array(face_stream).astype(np.float32)
#         if 'word' in self.modality:
#             retval['word'] = np.array(wrd_stream).astype(np.float32)
#         # Target
#         retval['Y'] = yval = np.reshape([1 if alab == 1 else 0 \
#             for alab in self.Y[talkid]],(1,-1)).astype(np.int64)
#         retval['ylabel'] = self.ylabel

#         return retval

# def collate_for_streamed(datalist):
#     '''
#     Pad and pack a list of datapoints obtained from TED_Rating_Streamed_Dataset
#     '''
#     alldata={}
#     Y_batch=[]
#     for j,akey in enumerate(['pose','face','word','Y']): 
#         if not akey in datalist[0]:
#             continue
#         sizelist = [np.size(item[akey],axis=0) for item in datalist]
#         idx = np.argsort([-asize for asize in sizelist])
#         sizelist = [sizelist[i] for i in idx]
#         alldata[akey] = []
#         for i in idx:
#             if akey == 'Y':
#                 Y_batch.append(datalist[i]['Y'])
#             else:
#                 dattemp = Variable(torch.from_numpy(datalist[i][akey]))
#                 alldata[akey]+=[dattemp]
#         if not akey=='Y':        
#             alldata[akey] = pad_sequence(alldata[akey])
#             alldata[akey] = pack_padded_sequence(alldata[akey],sizelist)
#     # Packing Y
#     Y_shaped = np.concatenate(Y_batch,axis=0)
#     alldata['Y'] = Variable(torch.from_numpy(Y_shaped))
#     return alldata

# def get_data_iter_streamed(dataset,batch_size=4,shuffle=True,
#         collate_fn=collate_for_streamed,
#         pin_memory=False):
#     dataiter = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
#         num_workers=mp.cpu_count()-1,collate_fn=collate_fn,
#         pin_memory=pin_memory)
#     return dataiter

def get_data_iter_simple(dataset,batch_size=4,shuffle=True,
        collate_fn=collate_and_pack_simple,
        pin_memory=False,gpuNum=-1):
    '''
    Uses pytorch's dataloader method to get an iterator over the dataset.
    The comes in packed condition
    '''
    dataiter = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
        num_workers=mp.cpu_count()-1,collate_fn=partial(collate_fn,
        gpuNum=gpuNum),pin_memory=pin_memory)
    return dataiter

def __buildalign2trmap__(alist,blist):
    '''
    Aligns two transcripts word by word and returns backpointer
    '''
    # initialization
    d = np.zeros((len(alist)+1,len(blist)+1))
    bp = np.zeros((len(alist)+1,len(blist)+1),dtype='i2,i2')
    if not (alist and blist):
        raise ValueError('Atleast one list is empty')
    d[:,0]=np.arange(np.size(d,axis=0))
    d[0,:]=np.arange(np.size(d,axis=1))        
    # Build up the distance and backpointer tables
    for i in range(1,len(alist)+1):
        for j in range(1,len(blist)+1):
            choices = [d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+2 \
                            if not alist[i-1]==blist[j-1] else d[i-1,j-1]]
            temp = np.argmin(choices)
            d[i,j] = choices[temp]
            bp[i,j] = [(i-1,j),(i,j-1),(i-1,j-1)][temp]
    # Build up the alignment from alist to blist
    align2trmap = {idx:-1 for idx in range(len(alist))}
    nd = (-1,-1)    
    while not (nd[0]==0 and nd[1]==0):
        p_nd = bp[nd[0],nd[1]]
        if d[nd[0],nd[1]] == d[p_nd[0],p_nd[1]]:
            align2trmap[p_nd[0]]=p_nd[1]
        nd = p_nd.copy()
    return align2trmap

def get_sent_boundary(atalk,computeAll=False):
    '''
    Returns the sentence boundary corresponding to each dependency tree in TED_meta.
    If the data is not available in TED_meta, it computes the data using the follwoing
    function: align_fave_transcript_to_word_boundary
    Once computed, it adds the info to TED_meta for faster processing next time.
    '''
    dir_meta = os.path.join(ted_data_path,'TED_meta')
    file_meta = os.path.join(dir_meta,str(atalk)+'.pkl')
    data_meta = cp.load(open(file_meta))

    dataexists = all(akey in data_meta for akey in ['wordlist_meta_sent', 'sent2wordmap',\
     'wordlist_bound_sent', 'alignmentmap', 'word_2_time_map', 'sent_2_time_map'])

    if not dataexists or computeAll:
        wordlist_meta_sent, sent2wordmap, wordlist_bound_sent, alignmentmap, \
          word_2_time_map, sent_2_time_map = align_fave_transcript_to_word_boundary(atalk)
        data_meta['wordlist_meta_sent']=wordlist_meta_sent
        data_meta['sent2wordmap']=sent2wordmap
        data_meta['wordlist_bound_sent']=wordlist_bound_sent
        data_meta['alignmentmap']=alignmentmap
        data_meta['word_2_time_map']=word_2_time_map
        data_meta['sent_2_time_map']=sent_2_time_map
        cp.dump(data_meta,open(file_meta,'wb'))
    return data_meta['sent_2_time_map']

def get_sent_boundary_new(atalk,computeAll=False):
    '''
    ========= FLAGGED FOR DEPRECATION. USE UTTERANCE LEVEL BOUNDARIES INSTEAD =========
    ========= SEE __check_fave_data_validity__ ========================================
    Returns the sentence boundary corresponding to each dependency tree in TED_meta.
    If the data is not available in TED_meta, it computes the data using the follwoing
    function: align_fave_transcript_to_word_boundary
    Once computed, it adds the info to TED_meta for faster processing next time.
    '''
    dir_meta = os.path.join(ted_data_path,'TED_meta')
    file_meta = os.path.join(dir_meta,str(atalk)+'.pkl')
    data_meta = cp.load(open(file_meta))

    dataexists = all(akey in data_meta for akey in ['utterances','utterances_starttime'])

    if not dataexists or computeAll:
        wordlist_meta_sent, sent2wordmap, wordlist_bound_sent, alignmentmap, \
          word_2_time_map, sent_2_time_map = align_fave_transcript_to_word_boundary(atalk)
        data_meta['wordlist_meta_sent']=wordlist_meta_sent
        data_meta['sent2wordmap']=sent2wordmap
        data_meta['wordlist_bound_sent']=wordlist_bound_sent
        data_meta['alignmentmap']=alignmentmap
        data_meta['word_2_time_map']=word_2_time_map
        data_meta['sent_2_time_map']=sent_2_time_map
        cp.dump(data_meta,open(file_meta,'wb'))
    return data_meta['sent_2_time_map']

def align_fave_transcript_to_word_boundary(atalk):
    '''
    ============ SET FOR DEPRECATION ===========
    Aligns each sentence in fave_style_transcript for TED_meta with the
    words in TED_feature_word_boundary. In returns the following:
    a) wordlist_meta_sent: list of words in fave_style_transcript
    b) sent2wordmap: For each sentence in fave_style_transcript the first and last word index
    c) wordlist_bound_sent: list of words in TED_feature_word_boundary
    d) alignmentmap: a map from index of wordlist_meta_sent to index of wordlist_bound_sent
    e) word_2_time_map: start and end time for index of wordlist_bound_sent
    f) sent_2_time_map: start and end time for each sentence in fave_style_transcript
    '''
    dir_meta = os.path.join(ted_data_path,'TED_meta')
    dir_boun = os.path.join(ted_data_path,'TED_feature_word_boundary')
    data_meta = cp.load(open(os.path.join(dir_meta,str(atalk)+'.pkl')))
    data_boun = cp.load(open(os.path.join(dir_boun,str(atalk)+'.pkl')))

    # Build a list of words from TED_feature_word_boundary (wordlist_bound_sent)
    word_2_time_map={}
    wordlist_bound_sent=[]
    n = len(data_boun['sentences'])
    for j in range(n):
        for aword in data_boun['sentences'][j]['word_time_boundaries']:
            if aword[0]=='sp' or aword[0]=='STOP':
                continue
            wordlist_bound_sent.append(aword[0])
            word_2_time_map[len(word_2_time_map)] = aword[1:]
    # Words from each sentence of TED_meta
    sent2wordmap = {}
    wordcount = 0
    frm = 0
    wordlist_meta_sent=[]
    alignmentmap={}
    sent_2_time_map={}
    # loop over each sentence in fave_style_transcript
    m = len(data_meta['dep_2_fave'])
    buffer = 5
    for i in range(m):
        # Pick each sentence from the fave_style_transcript, make a combined list of
        # the words (wordlist_meta_sent) and a map that points to the indices of the
        # words for each sentence
        paraID,sentID = data_meta['dep_2_fave'][i]
        meta_sent = data_meta['fave_style_transcript'][paraID]['sentences'][sentID].upper()
        meta_sent = re.sub(r'[^\w\d\s]',' ',meta_sent).strip()
        meta_sent = re.sub(r'\s+',' ',meta_sent)
        temp_wordlist = meta_sent.split()
        if not temp_wordlist:
            continue
        wordlist_meta_sent.extend(temp_wordlist)
        sent2wordmap[i]=(wordcount,wordcount+len(temp_wordlist))
        wordcount = len(wordlist_meta_sent)
        # Align words from fave_style_transcript with the list of words from 
        # TED_feature_word_boundary (wordlist_bound_sent)            
        upto = min(frm+len(temp_wordlist)+buffer,len(wordlist_bound_sent))
        buffer = 5
        if frm==upto:
            continue
        tempMap = __buildalign2trmap__(temp_wordlist,wordlist_bound_sent[frm:upto])
        if all(aval==-1 for aval in tempMap.values()):
            print temp_wordlist
            print wordlist_bound_sent[frm:upto]
            buffer = 5 + len(temp_wordlist)
            continue
        alignmentmap.update({k+sent2wordmap[i][0]:frm+v \
            for k,v in tempMap.items() if not v == -1})
        frm = alignmentmap.values()[-1]+1
        # Compute sentence boundary in time
        # find a valid left and right edge of the sentence
        left = sent2wordmap[i][0]
        right = sent2wordmap[i][1]
        while not left in alignmentmap or alignmentmap[left] == -1:
            left+=1
        while not right in alignmentmap or alignmentmap[right] == -1:
            right-=1
        if left>right:
            raise IOError('Sentence edges not found')
        left = alignmentmap[left]
        right = alignmentmap[right]
        sent_2_time_map[i]=(word_2_time_map[left][0],word_2_time_map[right][1])
    return wordlist_meta_sent, sent2wordmap, wordlist_bound_sent, \
        alignmentmap, word_2_time_map, sent_2_time_map

def __check_fave_data_validity__(atalk):
    '''
    Checks the validity of the fave style transcripts.
    This checking is done by matching the utterance level time-stamps with the
    sentence level time-stamps provided in the fave-style transcripts.
    For comparison, sentences must be lowercased, without any linefeed symbol, multiple
    whilespaces must be replaced by a single whitespace, and all the annotations
    must be removed e.g. (Laughter), (applause) etc.

    RESULT: Sentence boundaries in the Fave-style transcripts are erroreneous. It is
    better to use the utterance-level boundaries to find sentence boundaries.
    '''
    dir_meta = os.path.join(ted_data_path,'TED_meta')
    file_meta = os.path.join(dir_meta,str(atalk)+'.pkl')
    data = cp.load(open(file_meta))

    # Relevant regular expressions
    punct = re.compile('[.?!]$')
    lf = re.compile('\n')
    tagcatcher = re.compile('\(\w*\)')
    white_sp = re.compile('\s{2,}')


    sentence = ''
    sentList_utt = []
    original_utt = []
    # Getting sentences from utterances
    for anutt in data['utterances']:
        lfcleared = lf.sub(' ',anutt.lower())
        sentence+= ' '+lfcleared
        if punct.search(lfcleared):
            # found a sentence
            sentence = tagcatcher.sub('',sentence)
            sentence = white_sp.sub(' ',sentence).strip()
            sentList_utt.append(sentence)
            sentence = ''
    sentiter = iter(sentList_utt)
    # Getting sentences from fave style aligner
    for apara in data['fave_style_transcript']:
      for asent in apara['sentences']:
          nextsent = sentiter.next()
          asent = lf.sub(' ',asent.lower())
          asent = tagcatcher.sub('',asent)
          asent = white_sp.sub(' ',asent).strip()
          if not asent == nextsent:
            print 'Fave:',asent
            print 'utter:',nextsent
            import pdb; pdb.set_trace()  # breakpoint 80600ddd //

def __sent_len_hist__():
    import matplotlib.pyplot as plt
    bounds=[]                                   
    for i,atalk in enumerate(all_valid_talks):  
        print atalk,'...',                      
        bound = ttdf.get_sent_boundary(atalk)
        for akey in bound:                 
            sent_len=bound[akey][1]-bound[akey][0]
            print i,sent_len
            bounds.append(sent_len)
        print 'Done'
    plt.figure(0)
    plt.clf()
    plt.hist(bounds,50)
    plt.xlabel('Length of a sentence (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentence_length_hist.pdf')
    plt.figure(1)
    plt.clf()
    plt.hist(bounds,50)
    plt.yscale('log', nonposy='clip')
    plt.xlabel('Length of a sentence (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentence_length_hist_log.pdf')

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
    ttce.evaluate_clusters_pretty(X,comp)
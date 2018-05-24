import os
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

def get_minibatch_iter(dataset,minibatch_size,gpuNum,datakeys=['X','Y']):
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
                adata = {akey:variablize(dataset[idx[j]][akey],gpuNum)\
                    for akey in datakeys}
                batch.append(adata)
            yield batch
    else:
        for i in range(0,len(idx),minibatch_size):
            batch = []
            for j in range(i,min(len(idx),i+minibatch_size)):
                adata={}
                for akey in datakeys:
                    if akey=='X':
                        adata[akey]=[variablize(asent,gpuNum) for asent in dataset[idx[j]][akey]]
                    else:
                        adata[akey]=variablize(dataset[idx[j]][akey],gpuNum)
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


class TED_Rating_Averaged_Dataset(Dataset):
    """
    TED rating dataset. The pose and face values are averaged.
    Note: Input Semantics for LSTM
    The first axis is the sequence itself (T), the second indexes
    instances in the mini-batch (B), and the third indexes elements
    of the input (*).
    """

    def __init__(self, data_indices=lst_talks.all_valid_talks,firstThresh=50.,\
            secondThresh=50.,scale_rating=True,posedim=36,prosodydim=49,\
            facedim=44,worddim=300,modality=['word','pose','face','audio']):

        # get ratings
        self.Y,_,self.ylabel = binarized_ratings(firstThresh,\
            secondThresh,scale_rating)
        # Indices of the data in the dataset
        self.data_indices = list(set(data_indices).intersection(self.Y.keys()))

        # Other important information for the dataset
        print 'reading word vectors'
        # Reading the complete w2v dictionary
        self.wvec = read_crop_glove()
        print 'reading ratings'
        self.ratings = {akey:self.Y[akey] for akey in self.data_indices}
        self.posepath = os.path.join(ted_data_path,\
            'TED_feature_openpose/openpose_sentencewise_features/')
        self.facepath = os.path.join(ted_data_path,\
            'TED_feature_openface/openface_sentencewise_features/')
        self.sentpath = os.path.join(ted_data_path,\
            'TED_feature_sentence_boundary/')
        self.prospath = os.path.join(ted_data_path,\
            'TED_feature_prosody/per_sentence')
        self.modality = modality
        self.posedim = posedim
        self.prosdim = prosodydim
        self.facedim = facedim
        self.worddim = worddim

        self.dims = 0
        if 'pose' in self.modality:
            self.dims += posedim
        if 'face' in self.modality:
            self.dims += facedim
        if 'word' in self.modality:
            self.dims += worddim
        if 'audio' in self.modality:
            self.dims += prosodydim

        # Final check for the existence of all the files.
        all_available = set()
        print 'Checking indices'
        for atalk in self.data_indices:
            posefile = os.path.join(self.posepath,str(atalk)+'.pkl')
            facefile = os.path.join(self.facepath,str(atalk)+'.pkl')
            sentfile = os.path.join(self.sentpath,str(atalk)+'.pkl')
            prosfile = os.path.join(self.prospath,str(atalk)+'.pkl')
            if os.path.exists(posefile) and os.path.exists(facefile) \
                and os.path.exists(sentfile) and os.path.exists(prosfile):
                all_available.add(atalk)
        # Remove indices that are not all available
        self.data_indices = list(all_available.intersection(self.data_indices))
        print 'Dataset Ready'

        
    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        '''
        Given a data index (not videoid), get the corresponding data
        '''
        talkid = self.data_indices[idx]
        posedat = cp.load(open(os.path.join(self.posepath,str(talkid)+'.pkl')))
        facedat = cp.load(open(os.path.join(self.facepath,str(talkid)+'.pkl')))
        facedat = facedat[0]
        sentdat = cp.load(open(os.path.join(self.sentpath,str(talkid)+'.pkl')))
        sentdat = sentdat['sentences']
        prosdat,_ = read_prosody_per_sentence(talkid)

        outvec=[]
        # Second best candidate for data parallelization. CPU multithreading
        for asent,aface,apose,apros in izip_longest(sentdat,facedat,\
                posedat,prosdat,fillvalue=[]):
            # Average pose per sentence
            if 'pose' in self.modality:
                if not apose:
                    apose = np.zeros(self.posedim)
                else:
                    assert self.posedim == np.size(apose,axis=1)
                    apose = np.nanmean(np.nan_to_num(apose),axis=0)
            # Average face per sentence
            if 'face' in self.modality:
                if not aface:
                    aface = np.zeros(self.facedim)
                else:
                    assert self.facedim == np.size(aface,axis=1)
                    aface = np.nanmean(np.nan_to_num(aface),axis=0)
            # Average, std, skew, kartosys of prosody per sentence
            if 'audio' in self.modality:
                if not apros:
                    apros = np.zeros(self.prosdim)
                else:
                    apros = np.nan_to_num(apros)

            # Concatenate pose and face and prosody
            if 'pose' in self.modality and 'face' in self.modality:
                nonverbal = np.concatenate((apose,aface,apros)).tolist()
            elif 'pose' in self.modality:
                nonverbal = apose
            elif 'face' in self.modality:
                nonverbal = aface
            elif 'audio' in self.modality:
                nonverbal = apros
            else:
                nonverbal = []

            # Take all words in a sentence
            if 'word' in self.modality:
                allwords=[]
                nullvec = [0 for i in range(self.worddim)]
                for aword in word_tokenize(asent['sentence']):
                    if aword in self.wvec:
                        allwords.append(self.wvec[aword]+nonverbal)
                    else:
                        allwords.append(nullvec+nonverbal)
                outvec.extend(allwords)
            else:
                outvec.extend([nonverbal])
        # Note: This is T x * (not T x B (batch = 1) x *
        # Because the collate function needs in this form for
        # pad_sequence
        xval = np.array(outvec).astype(np.float32)

        yval = np.reshape([1 if alab == 1 else 0 for alab in \
            self.Y[talkid]],(1,-1)).astype(np.int64)
        return {'X':xval,'Y':yval,'ylabel':self.ylabel}

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
    """TED rating dataset.
    Only the word modality is used. It will pass the indices of the
    word-vectors for each word in the transcript. For out of vocabulary
    word, it will put -1 as the index. For compound words (e.g. modern-world),
    it will juxtapose the indices of the constituent words.
    
    Note: Input Semantics for LSTM
    The first axis is the sequence itself (T), the second indexes
    instances in the mini-batch (B), and the third indexes elements
    of the input (*).
    """

    def __init__(self, data_indices=lst_talks.all_valid_talks,firstThresh=50.,\
            secondThresh=50.,scale_rating=True,flatten_sentence=False):

        # get ratings
        self.Y,_,self.ylabel = binarized_ratings(firstThresh,\
            secondThresh,scale_rating)
        # Indices of the data in the dataset
        self.data_indices = list(set(data_indices).intersection(self.Y.keys()))
        ################ DEBUG * REMOVE ###############
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # self.data_indices = self.data_indices[:10]
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        ###############################################
        print 'reading ratings'
        self.ratings = {akey:self.Y[akey] for akey in self.data_indices}

        # Other important information for the dataset
        print 'reading word vectors'
        self.flatten_sentence=flatten_sentence
        self.wvec_map = wvec_index_maker()
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
            xval = np.array(outvec).astype(np.int64).reshape(-1,1)
        else:
            xval=[np.array(asent).astype(np.int64).reshape(-1,1) \
                for asent in outvec]
        # Process the labels
        yval = np.reshape([1 if alab == 1 else 0 for alab in \
            self.Y[talkid]],(-1,1)).astype(np.float32)
        return {'X':xval,'Y':yval,'ylabel':self.ylabel}

    
class wvec_index_maker():
    '''
    A class to convert the transcript of any talk to a list of word-vector indices.
    It takes care of the word pre-processing issues. For convenience, it also saves
    the wordvector values (w2v_vals) as a matrix (easier gputization). The rows of
    this matrix is consistent with the w2v_indices. Initialize it only once because
    it loads the w2v dictionary while initializing.
    '''
    def __init__(self):
        # Reading the complete w2v dictionary
        wvec = read_crop_glove()
        self.w2v_indices = {akey:i for i,akey in enumerate(wvec)}
        self.w2v_vals = np.array(wvec.values()).astype(np.float32)
        self.dims = len(wvec['the'])
    
    def __call__(self,talkid,flatten_sentence=True):
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
                        allidx.extend(self.word2idx(wordlist))
                    else:
                        allidx.append(self.word2idx(wordlist))
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
        return allidx

class TED_Rating_Streamed_Dataset(TED_Rating_Averaged_Dataset):
    """
    TED rating dataset. The word, pose and face values are
    sent as three different streams.
    
    Note: Input Semantics for LSTM
    The first axis is the sequence itself (T), the second indexes
    instances in the mini-batch (B), and the third indexes elements
    of the input (*).
    """
    def __getitem__(self, idx):
        '''
        Given a data index (not videoid), get the corresponding data
        '''
        talkid = self.data_indices[idx]
        posedat = cp.load(open(os.path.join(self.posepath,str(talkid)+'.pkl')))
        facedat = cp.load(open(os.path.join(self.facepath,str(talkid)+'.pkl')))
        facedat = facedat[0]
        sentdat = cp.load(open(os.path.join(self.sentpath,str(talkid)+'.pkl')))
        sentdat = sentdat['sentences']

        pose_stream=[]
        face_stream=[]
        wrd_stream=[]
        nullwvec = [0 for i in range(self.worddim)]
        nullpvec = [[0 for i in range(self.posedim)]]
        nullfvec = [[0 for i in range(self.facedim)]]
        retval = {}

        for asent,aface,apose in izip_longest(sentdat,facedat,posedat,\
                fillvalue=[]):
            # Pose
            if not apose:
                pose_stream.extend(nullpvec)
            else:
                pose_stream.extend(np.nan_to_num(apose).tolist())
            # Face
            if not aface:
                face_stream.extend(nullfvec)
            else:
                face_stream.extend(np.nan_to_num(aface).tolist())
            # word
            allwords=[]
            for aword in word_tokenize(asent['sentence']):
                if aword in self.wvec:
                    allwords.append(self.wvec[aword])
                else:
                    allwords.append(nullwvec)
            wrd_stream.extend(allwords)
        # Selectively add modalities
        if 'pose' in self.modality:
            retval['pose'] = np.array(pose_stream).astype(np.float32)
        if 'face' in self.modality:
            retval['face'] = np.array(face_stream).astype(np.float32)
        if 'word' in self.modality:
            retval['word'] = np.array(wrd_stream).astype(np.float32)
        # Target
        retval['Y'] = yval = np.reshape([1 if alab == 1 else 0 \
            for alab in self.Y[talkid]],(1,-1)).astype(np.int64)
        retval['ylabel'] = self.ylabel

        return retval

def collate_for_streamed(datalist):
    '''
    Pad and pack a list of datapoints obtained from TED_Rating_Streamed_Dataset
    '''
    alldata={}
    Y_batch=[]
    for j,akey in enumerate(['pose','face','word','Y']): 
        if not akey in datalist[0]:
            continue
        sizelist = [np.size(item[akey],axis=0) for item in datalist]
        idx = np.argsort([-asize for asize in sizelist])
        sizelist = [sizelist[i] for i in idx]
        alldata[akey] = []
        for i in idx:
            if akey == 'Y':
                Y_batch.append(datalist[i]['Y'])
            else:
                dattemp = Variable(torch.from_numpy(datalist[i][akey]))
                alldata[akey]+=[dattemp]
        if not akey=='Y':        
            alldata[akey] = pad_sequence(alldata[akey])
            alldata[akey] = pack_padded_sequence(alldata[akey],sizelist)
    # Packing Y
    Y_shaped = np.concatenate(Y_batch,axis=0)
    alldata['Y'] = Variable(torch.from_numpy(Y_shaped))
    return alldata

def get_data_iter_streamed(dataset,batch_size=4,shuffle=True,
        collate_fn=collate_for_streamed,
        pin_memory=False):
    dataiter = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
        num_workers=mp.cpu_count()-1,collate_fn=collate_fn,
        pin_memory=pin_memory)
    return dataiter

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


import os
import json
import glob
import cPickle as cp
import numpy as np
from list_of_talks import all_valid_talks
from TED_data_location import ted_data_path, wordvec_path

def split_train_test(train_ratio=0.7):
    '''
    Split the list of valid talk id's in training and test sets.
    '''
    m = len(all_valid_talks)
    train = np.random.choice(all_valid_talks,int(train_ratio*m),\
        replace=False).tolist()
    test = list(set(all_valid_talks) - set(train))
    return train,test

def generate_dep_tag(talk_id,dep_type='recur',tag_type='{LG}'):
    '''
    Given a talk_id, it generates the dependency tree and the tag information.
    The dep_type can take two values: 'recur' means dependency tree in
    recursive format, 'conll' means conll format.
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

def get_dep_rating(talk_id,dep_type='recur'):
    ''' 
    Given a talk_id, it generates the meta information and returns
    the dependency tree and the corresponding ground truth ratings.
    The dep_type can take three values: 'recur' means dependency tree in
    recursive format, 'conll' means conll format. If dep_type is assigned
    to 'both', the generator will generate both kinds of the dependency
    trees.
    '''
    filename = os.path.join(ted_data_path,'TED_meta/'+str(talk_id)+'.pkl')
    data = cp.load(open(filename))
    alldat = []
    ratedict = data['talk_meta']['ratings']
    ratedict_processed = {akey:float(ratedict[akey])/float(ratedict['total_count'])\
            for akey in sorted(ratedict) if not akey=='total_count'}

    for (i,j),adep,bdep in zip(data['dep_2_fave'],\
            data['dep_trees_recur'],\
            data['dep_trees_conll']):
        # Sentence
        sent = data['fave_style_transcript'][i]['sentences'][j]
        if dep_type == 'both':
            alldat.append([adep,bdep,(i,j),sent])
        elif dep_type == 'recur':
            alldat.append([adep,(i,j),sent])
        elif dep_type == 'conll':
            alldat.append([bdep,(i,j),sent])
        else:
            raise IOError('wrong value for dep_type')
    return alldat,ratedict_processed

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
    ted_meta_unprocessed_path = os.path.join(ted_data_path,'TED_meta_unprocessed/')
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
                    # If the word contains hyphen or period, split them and add
                    # them as well. Also add a hypen and periodless version because
                    # these things often cause confusions. This will be required
                    # when the engine would try to resolve unfound words.
                    if '-' in aword:
                        ted_vocab.update(aword.split('-'))
                        ted_vocab.add(aword.replace('-',''))
                    if '.' in aword:
                        ted_vocab.update(aword.split('.'))
                        ted_vocab.add(aword.replace('.',''))
        except Exception as e:
            print e
            print 'Moving the meta file {} to {}'.format(atalk,ted_meta_unprocessed_path)
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

if __name__=='__main__':
    print 'preparing data'
    print 'please make sure that the TED_data_location.py file is updated'+\
        'with the correct data location'
    print 'Preparing the glove dictionary'
    crop_glove_dictionary()


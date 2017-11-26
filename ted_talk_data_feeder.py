import os
import json
import glob
import cPickle as cp
from TED_data_location import ted_data_path, wordvec_path

def gen_dep_tag_data(talk_id,dep_type='recur',tag_type='{LG}'):
    '''
    Given a talk_id, it generates the dependency tree and the tag information.
    The dep_type can take three values: 'recur' means dependency tree in
    recursive format, 'conll' means conll format. If dep_type is assigned
    to 'both', the generator will generate both kinds of the dependency
    trees.
    The tag_type takes two values: '{LG}' means laughter followed by the
    sentence. {NS} means applause
    '''
    filename = os.path.join(ted_data_path,'TED_meta/'+str(talk_id)+'.pkl')
    data = cp.load(open(filename))
    if not dep_type=='both':
        for (i,j),adep in zip(data['dep_2_fave'],\
                data['dep_trees_'+dep_type]):
            # Tag label
            label = True if \
                    data['fave_style_transcript'][i]['labels'][j]==tag_type\
                    else False
            # sentence
            sent = data['fave_style_transcript'][i]['sentences'][j]
            yield adep,label,(i,j),sent
    else:
        for (i,j),adep,bdep in zip(data['dep_2_fave'],\
                data['dep_trees_recur'],data['dep_trees_conll']):
            # Label
            label = True if \
                    data['fave_style_transcript'][i]['labels'][j]==tag_type\
                    else False
            # Sentence
            sent = data['fave_style_transcript'][i]['sentences'][j]
            yield adep,bdep,label,(i,j),sent

def dep_rating_data(talk_id,dep_type='recur'):
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
    for atalk in glob.glob(ted_meta_path):
        print atalk
        data = cp.load(open(atalk))
        for aline in data['dep_trees_conll']:
            for aword in [anode[0].encode('ascii','ignore').strip()\
                    for anode in aline]:
                if not aword in ted_vocab:
                    ted_vocab.add(aword)
    return ted_vocab

def crop_glove_dictionary():
    '''
    This function creates a reduced version of the glove word2vec dictionary 
    by deleting all the words that did not appear in the ted talk vocabulary.
    '''
    print 'building ted vocab'
    ted_vocab = build_ted_vocab()
    print 'TED Vocabulary size',len(ted_vocab)
    with open(wordvec_path) as fin:
        with open(wordvec_path+'_cropped','wb') as fout:
            rows=[]
            print 'reading glove w2v'
            for i,aline in enumerate(fin):
                if i%100000==0:
                    print i
                splt_line = aline.strip().split()
                # read the line words in unicode
                if splt_line[0].strip() in ted_vocab:
                    rows.append(splt_line)
                    ted_vocab.remove(splt_line[0])
            print ted_vocab
            print '# of words from TED found in glove word2vec',len(rows)
            print 'writing cropped glove w2v dict'
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
            retdict[splt_line[0]]=map(float,splt_line[1:])
    return retdict

def traverse(atree,level=0):
    '''
    Recursively travarses and prints a dependency tree
    '''
    for anode in atree:
        if type(anode)==str or type(anode)==unicode:
            print '-'*level,anode
        elif type(anode)==list:
            travarse(anode,level+1)

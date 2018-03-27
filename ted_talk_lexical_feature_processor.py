import os
import csv
import cPickle as cp
from TED_data_location import ted_data_path
from list_of_talks import all_valid_talks
from nltk.tokenize import word_tokenize

def ReadLIWCDictionary(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    
    dic = {}

    for line in lines:
        parts = line.lstrip().rstrip().split("\t")

        values = list()
        for i in range(1, len(parts)):
#            print(parts[0], parts[i])
            values.append(int(parts[i]))

        dic[parts[0]] = values

    return dic

def ReadLIWCCategories(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    categories = lines[0].split("\r")
    catdic = {}
    
    for cat in categories:
        catparts = cat.split("\t")
        catdic[int(catparts[0])] = catparts[1]
    return catdic

liwcpath = os.path.join(ted_data_path,'misc/')
LIWCDic = ReadLIWCDictionary(os.path.join(liwcpath,'liwcdic2007.dic'))
categories = ReadLIWCCategories(os.path.join(liwcpath,'liwccat2007.txt'))

def match(word,LIWCDic=LIWCDic):
    if word in LIWCDic:
        return LIWCDic[word]
    
    for i in range(1,len(word)):
        key = word[:i] + "*"
        if key in LIWCDic:
            return LIWCDic[key]
    return list()

def feat(wrdlist,LIWCDic=LIWCDic,cats=categories):
    feat_count={cats[acat]:0 for acat in cats}
    m = float(len(wrdlist))
    for awrd in wrdlist:
        for acat in match(awrd):
            if acat in cats:
                # Word LIWC categories normalized by word count
                feat_count[cats[acat]]+=1./m
    return feat_count

def prepare_lexical_feat(talklist=all_valid_talks,
    featurefile = 'misc/lexical.csv'):
    '''
    Reads, prepares the lexical features and writes in featurefile
    '''
    with open(os.path.join(ted_data_path,featurefile),'wb') as fout:
        writer = None
        for atalk in talklist:
            print 'Processing Lexical Features of',atalk
            pklname = os.path.join(ted_data_path,'TED_meta/'+str(atalk)+'.pkl')
            if not os.path.exists(pklname):
                print 'Not found',atalk
                continue
            data = cp.load(open(pklname))
            wrds = word_tokenize(' '.join([aline for apara in \
                data['talk_transcript'] for aline in apara]))
            features = feat(wrds)
            # Read the labels for the first time
            if not writer:
                writer = csv.DictWriter(fout,['TalkID']+\
                    sorted(features.keys()))
                writer.writeheader()
            features['TalkID'] = atalk
            writer.writerow(features)
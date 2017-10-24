from TED_data_location import ted_data_path
import cPickle as cp
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from multiprocessing import Process
from itertools import izip
import json
import stat
import glob 
import os
import re

'''
This module extracts the dependency tree for each sentence of the transcript
located inside the TED_meta pickle file.
It also updates the pickle file (in place) with the dependency information.
It must run from within the brianlow/syntaxnet docker container default path.
'''
def generate_transcript(data_path):
    '''
    Given the path to TED_meta folder, generates all the
    transcripts and the corresponding talk id.
    '''
    pkl_path = os.path.join(data_path,'TED_meta/*.pkl')
    for pkl_file in glob.glob(pkl_path):
        pkl_id = int(os.path.split(pkl_file)[-1].split('.')[0])
        data = cp.load(open(pkl_file))
        txt = ' '.join([aline.encode('ascii','ignore').lower()\
                for apara in data['talk_transcript'] for aline in apara])
	# Remove the parenthesized items (tags)
        txt = re.sub('\([a-zA-Z]*?\)','',txt)
        # Remove everything except alpha neumeric, space and punctiation
        # after sentence tokenization. Then join the sentences with newline.
        txt = '\n'.join([re.sub('[^\w\ ,.!?;\']','',asent) for asent in sent_tokenize(txt)])
        yield pkl_id,txt

def get_dep_tree(txt,
    shellname='syntaxnet/demo.sh'):
    '''
    Parses a query using syntaxnet.
    '''
    allparsetreelist=[]
    # get the parse tree
    output = os.popen('echo '+'"'+txt+'"'+' | '+shellname).read().split('\n')
    return output

def make_new_shell():
    '''
    Prepare a shell for conll style dependency tree
    This function must run within the docker image of syntaxnet
    '''
    with open('syntaxnet/demo.sh') as f:
        txt = f.read()
    with open('syntaxnet/demo_conll.sh','wb') as f:
        f.write(txt[:-107]+'\n')
    st = os.stat('syntaxnet/demo_conll.sh')
    os.chmod('syntaxnet/demo_conll.sh', st.st_mode | stat.S_IEXEC)

def segment_gen(inp_gen):
    '''
    Segments the stream from the output of syntaxnet into
    one sentence and one parse tree at a time. The parse tree
    is given in a json format.
    :param inp_gen: input iterator 
    '''
    retval = ''
    parsetree=''
    currlevel=-1
    count = 0
    for inp in inp_gen:
        # Transforming the tree with states "Input", "Parse", and
        # the normal tree parsing.
        if inp.startswith('Input'):
            # if there is something in retval from previous iterations
            if retval:
                # Close off the tree 
                retval+=parsetree+']'*(currlevel+2)
                yield retval
                # Reset the value
                retval=''
            # There is nothing from previous iterations, so start making
            retval+=inp[6:].strip() +'\t'
        elif inp.startswith('Parse'):
            # start of the parse tree
            parsetree=''
        elif not inp:
            # if the input is empty, just skip it
            continue
        else:
            parse_out,currlevel=jsonify_tree(inp,currlevel)
            # Debug
            # print inp,parse_out,currlevel
            parsetree+=parse_out
    if retval and parsetree:
        # Close off the last tree
        retval+=parsetree+']'*(currlevel+2)
        yield retval
    

def segment_gen_conll(inp_gen):
    '''
    similar to segment_gen, but works on conll style parse tree
    '''
    aparse=[]
    for inp in inp_gen:
        if not inp:
            yield json.dumps(aparse)
            aparse=[]
        else:
            aparse.append(inp.split('\t')[1:])

def jsonify_tree(inp,currlevel):
    '''
    Converts from syntaxnet tree structure to json tree structure.
    '''
    nxtlevel = inp.find('+--')/4
    if nxtlevel == -1:
        # Root Node
        return '[ '+'"'+ inp+'"',-1
    elif nxtlevel==currlevel+1:
        # Subtree of previous node
        return ', [ '+'"'+inp[nxtlevel*4+4:].strip()+'"',nxtlevel
    elif nxtlevel==currlevel:
        # Another node in the same level of the tree
        return ', '+'"'+inp[nxtlevel*4+4:].strip()+'"',nxtlevel
    elif nxtlevel<currlevel:
        # At least one subtree finished
        leveljump = currlevel - nxtlevel
        return ']'*leveljump+','+'"'+inp[nxtlevel*4+4:].strip()+'"',nxtlevel
    else:
        # nxtlevel>currlevel+1 
        # Impossible situation. Something is wrong
        raise IOError('More than one level jump forward. At least one tree node must be missing.')

def pipeline(startid,endid):
    # Create a shell for conll style dependency tree
    make_new_shell()
    # Iterator for generating the transcript from the input folder
    transc_iter = generate_transcript(ted_data_path)
    # Iterate over the transcripts
    for pkl_id, atranscript in transc_iter:
        # skip out of range pkl files
        if pkl_id < startid or pkl_id >= endid:
            continue
	# Get an iterator for the recursive style dependency tree
        dtree = get_dep_tree(atranscript)
	dtree_iter = segment_gen(dtree)
        # Get an iterator for the conll style dependency tree
        dtree_conll = get_dep_tree(atranscript,shellname='syntaxnet/demo_conll.sh')
	dtree_conll_iter = segment_gen_conll(dtree_conll)
        # Make list out of the iterators
        allsent=[]
        allparse=[]
        allparse_conll=[]
        for dtree,dtree_conll in izip(dtree_iter,dtree_conll_iter):
            sentence,parse = dtree.split('\t')
            allsent.append(sentence)
            allparse.append(parse)
            allparse_conll.append(dtree_conll)
	# Open Pickle file
	pkl_filename = os.path.join(ted_data_path,'TED_meta/'+str(pkl_id)+'.pkl')
        pkl_outfilename = os.path.join(ted_data_path,'TED_meta_with_dependency/'+str(pkl_id)+'.pkl')
        data_in = cp.load(open(pkl_filename))
        data_in['trans_dep_sentences']=allsent
        data_in['trans_dep_trees_rec']=allparse
        data_in['trans_dep_trees_conll']=allparse_conll
        # Save updated meta
        with open(pkl_outfilename,'wb') as f:
            cp.dump(data_in,f)

if __name__=='__main__':
    p1 = Process(target=pipeline,args=(1,725))
    p1.start()
    p2 = Process(target=pipeline,args=(725,1450))
    p2.start()
    p3 = Process(target=pipeline,args=(1450,2175))
    p3.start()
    p4 = Process(target=pipeline,args=(2175,2900))
    p4.start()
 










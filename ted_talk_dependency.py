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
import csv

'''
This module extracts the dependency tree for each sentence of the transcript
located inside the TED_meta pickle file.
It also updates the pickle file (in place) with the dependency information.
It must run from within the brianlow/syntaxnet docker container default path.
'''

path,filename = os.path.split(os.path.realpath(__file__))
names = cp.load(open(os.path.join(path,'names.pkl')))['names']

def split_speakername(txt,speakername):
    '''
    Separate the speaker name and the text
    '''
    temp=[]
    for aname in speakername:
        temp.append(aname.replace('_',' '))
        aname = re.split('[_ ]',aname)
        temp.append(aname[0][0]+aname[-1][0])
        temp.append(aname[0][0]+aname[-1])
        temp.append(aname[0]+aname[-1][0])
        temp.append(''.join([part[0] for part in aname]))
    speakername.extend(temp)
    txt = txt.lower()
    if ':' in txt:
        txtsplt = txt.split(':')
        candidate = txtsplt[0].strip()
        txtrest = ':'.join(txtsplt[1:])[1:]
        if candidate.count(' ')< 3 and (candidate in speakername or \
           any([aname in speakername for aname in candidate.split()]) or \
           re.match('narrator|moderator|host|girl[\w ]*|boy[\w ]*|'+\
                   'man[\w ]*|woman[\w ]*|audience|video|audio|'+\
                   'voice[\w ]*|child|lawyer|server|customer|'+\
                   'interpreter|student',candidate) or \
           candidate.count(' ')==0 or\
           any([aname in names for aname in candidate.split()])):
               return candidate.replace(' ','_'),txtrest
    if speakername:
        return speakername[0].replace(' ','_'),txt
    else:
        return 'no_name',txt

def process_trans_fave(pklfile):
    '''
    Process the transcript to FAVE format. This format has the following info
    for each column:
    1. Speaker ID
    2. Speaker Name
    3. Beginning Time
    4. End Time
    5. Sentences within the timeframe with tags ({LG} for laughter
       and {NS} for applause)
       
    '''
    data = cp.load(open(pklfile))
    alldata = json.loads(data['talk_meta']['alldata_JSON'])
    # The transcripts must be preprocessed and simplified
    data['talk_transcript'] = [[re.sub('[^\w\s\(\)\'\.\,\?\!\-\:]*','',\
            anitem.encode('ascii','ignore')) \
            for anitem in apara] for apara in data['talk_transcript']]
    m = len(data['talk_transcript'])
    rows=[]
    for i in range(m):
        rowdict={}
        # Time
        rowdict['beg_time'] = float(data['transcript_micsec'][i])/1000.
        if i==m-1:
            rowdict['end_time'] = float(data['talk_meta']['vidlen'])
        else:
            rowdict['end_time'] = float(data['transcript_micsec'][i+1])/1000.
        # speaker name identification
        firstline = data['talk_transcript'][i][0]
        allspeakers = [aspeaker['slug'].encode('ascii','ignore').lower()\
                for aspeaker in alldata['speakers']]
        speaker,txt = split_speakername(firstline,allspeakers)
        data['talk_transcript'][i][0] = txt
        rowdict['speaker_id']=speaker
        rowdict['speaker_name']=speaker
        # Process sentences and tags
        txt = ' '.join(data['talk_transcript'][i])
        # Clean all tags except laughter and applause
        txt = re.sub('\([Ll]aug\w*\)','{LG}',txt)
        txt = re.sub('\([Aa]pplause\)|\([Cc]heer\w*\)|\([Cc]lap\w*\)','{NS}',txt)
        txt = re.sub('\([\w ]*\)','',txt)
        lines = sent_tokenize(txt)
        n = len(lines)
        rowdict['sentences']=[]
        rowdict['labels']=[]
        for j in range(n):
            tags_st = re.findall('^{LG}|^{NS}|^{LG}\s*{NS}|^{NS}\s*{LS}',\
                    lines[j].strip())
            tags_mid = re.match('{LG}|{NS}',\
                    re.sub('^{LG}|^{NS}|^{LG}\s*{NS}|^{NS}\s*{LS}','',\
                    lines[j].strip()))
            if tags_st:
                # If the tags are in the beginning, add tag to previous sentence
                if j == 0 and i > 0 and len(rows)>0:
                    # add tag to last line of previous row
                    try:
                        rows[-1]['labels'][-1]+=''.join(tags_st)
                    except IndexError:
                        pass
                if j > 0:
                    # add tag to last line (so far) of this row
                    try:
                        rowdict['labels'][-1]+=''.join(tags_st)
                    except IndexError:
                        pass
            if not tags_mid:
                # If there is no mid or end tag leave the label field empty
                rowdict['labels'].append('')
            else:
                # otherwise add the mid or end tags to this row
                rowdict['labels'].append(''.join(tags_mid))
            # Add the sentence to this row
            rowdict['sentences'].append(lines[j].strip())
        # Add the row to list of rows
        rows.append(rowdict)
    return rows

def write_trans_fave(rows,filestream):
    '''
    Given a transcript processed for fave format, it will write the transcript
    in a csv file following the format.
    '''
    writer = csv.DictWriter(filestream,fieldnames=['speaker_id','speaker_name',\
            'beg_time','end_time','sentences'])
    writer.writeheader()
    for arow in rows:
        arow['sentences'] = ' '.join(arow['sentences'])
        del arow['labels']
        writer.writerow(arow)

def process_trans_syntaxnet(rows):
    '''
    Creates a new-line separated list of sentences from the fave transcript
    in order to pass through syntaxnet. Cleans the empty lines, tag only lines
    and changes the case to lower. Keeps a backpointer to trace back the
    original sentence once the dependency tree is obtained.
    '''
    bp=[]
    txt=[]
    for j,arow in enumerate(rows):
        for i,asent in enumerate(arow['sentences']):
            sent = re.sub('{LG}|{NS}','',asent).strip().lower()
            if sent:
                txt.append(sent)
                bp.append((j,i))
    return '\n'.join(txt),bp

def generate_transcript(data_path):
    '''
    Given the path to TED_meta folder, generates all the
    transcripts and the corresponding talk id.
    '''
    pkl_path = os.path.join(data_path,'TED_meta/*.pkl')
    for pkl_file in glob.glob(pkl_path):
        rows = process_trans_fave(pkl_file)
        txt,bp = process_trans_syntaxnet(rows)
        txt=txt.replace('"','')
        pkl_id = int(os.path.split(pkl_file)[-1].split('.')[0])
        yield pkl_id,txt,rows,bp

def get_dep_tree(txt,shellname='syntaxnet/demo.sh'):
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
    for pkl_id,atranscript,rows,bp in transc_iter:
        # skip out of range pkl files
        if pkl_id < startid or pkl_id >= endid:
            continue
        pkl_filename = os.path.join(ted_data_path,'TED_meta/'+str(pkl_id)+'.pkl')
        pkl_outfilename = os.path.join(ted_data_path,\
                'TED_meta_with_dependency/'+str(pkl_id)+'.pkl')
        # Skip if already exists
        if os.path.exists(pkl_outfilename):
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
            allparse.append(json.loads(parse))
            allparse_conll.append(json.loads(dtree_conll))
	# Open Pickle file
	data_in = cp.load(open(pkl_filename))
        # Clean old junk if they exist
        if 'trans_dep_sentences' in data_in:
            del data_in['trans_dep_sentences']
        if 'trans_dep_trees_rec' in data_in:
            del data_in['trans_dep_trees_rec']
        if 'trans_dep_trees_conll' in data_in:
            del data_in['trans_dep_trees_conll']
        # Add new data
        data_in['dep_trees_conll']=allparse_conll
        data_in['dep_trees_recur']=allparse
        data_in['fave_style_transcript']=rows
        data_in['dep_2_fave']=bp
        # Save updated meta
        with open(pkl_outfilename,'wb') as f:
            cp.dump(data_in,f)

if __name__=='__main__':
    # Debug
    #pipeline(1,10)
    p1 = Process(target=pipeline,args=(1,725))
    p1.start()
    p2 = Process(target=pipeline,args=(725,1450))
    p2.start()
    p3 = Process(target=pipeline,args=(1450,2175))
    p3.start()
    p4 = Process(target=pipeline,args=(2175,2900))
    p4.start()
 


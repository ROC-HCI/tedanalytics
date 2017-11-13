import os
import cPickle as cp
from TED_data_location import ted_data_path

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



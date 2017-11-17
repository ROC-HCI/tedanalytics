import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ted_talk_data_feeder as ttdf
from torch.nn.parameter import Parameter
from TED_data_location import ted_data_path, wordvec_path

class Embed

class SyntacticSemanticEngine(nn.Module):
    '''
    Syntactic Semantic Engine is a new model for representing a dependency
    tree along with the distributed representations of the corresponding words.
    '''
    def __init__(self,posvoclen,depvoclen,sensedim=300):
        super(SyntacticSemanticEngine,self).__init__()
        # List of dimensions
        self.s = sensedim
        self.p = posvoclen
        self.d = depvoclen
        pass
        

    def encodetree(self,tree,dep_pos_dict,glove_voc):
        encodedtree=
        for anode in tree:
            if type(anode) == list:
                encodedtree.append(encodetree(anode,dep_pos_dict,glove_voc))
            elif type(anode) == unicode:
                splt_node = anode.encode('ascii','ignore').split()
                if splt_node[0] in glove_voc:
                    encodedtree.append(glove_voc[splt_node[0]])
                    pass





import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

from embeddings import VocabEmbeddings


class DST(nn.Module):
    """ Dialogue State Tracking Model
    """
    def __init__(self, embed_dim, sentence_hidden_dim, hierarchial_hidden_dim, da_hidden_dim, da_embed_size, 
                    ff_hidden_dim, batch_size, num_slots):
        super(DST, self).__init__()

        # instantiate candidate_embedding
        self.candidate_embeddings = VocabEmbeddings(embed_dim)
        self.sentence_encoder = SentenceBiLSTM(sentence_hidden_dim, embed_dim, self.candidate_embeddings, batch_size)
        self.hierarchial_encoder = HierarchicalLSTM(sentence_hidden_dim, hierarchial_hidden_dim)

        # TODO: instantiate DialogueActsLSTM

        # context_dim = | [E_i; Z_i; A_i; C_ij] | where E_i = encoded utterance, Z_i = encoding of past user utterances, 
        #                                               A_i = system action utterances, C_ij = candidate encoding
        self.context_cand_dim = embed_dim + 2 * (sentence_hidden_dim) + hierarchial_hidden_dim + da_hidden_dim
        self.classification_net = ClassificationNet(self.context_cand_dim, ff_hidden_dim, num_slots)

    def forward(self, dialogue, slot_values):
        """ Forward pass for an entire dialogue
            @param dialogue (Dict): contains turn by turn conversation information such as user utterances,
                            system dialogue acts, and ground truth slot values
            @param slot_values (Tensor): specifies relevant slot values for a particular domain
            @returns filled_slots (Dict): mapping of slots to values
        """
        pass 

    def forward_turn(self, user_utterance, past_user_utterances, dialogue_acts, current_state):
        """ Forward pass for a user turn in a given dialogue
            @param user_utterance (String): Current user utterance
            @param past_user_utterances (Tensor): Tensor of past encoded user utterances
            @param dialogue_acts (Tensor): List of dialogue acts (represented as strings) in the format
                                        dialogue_act(slot_type)
            @param current_state: current Dialogue state
            @returns current_state, past_user_utterances, loss: returns updated current_state, updates past_user_utterances
                                to include the newly encoded user utterance and returns the loss over all candidates for the turn    
        """
        pass

    def train_one_batch(self, dialogues):
        """ Trains model over a single batch
        """
        pass

class SentenceBiLSTM(nn.Module): 
    """ BiLSTM used to encode individual user utterances 
    """
    def __init__(self, hidden_dim, embed_dim, candidate_encoder, batch_size):
        """ Init SentenceBiLSTM

        @param embed_dim (int): Embedding size (dimensionality)
        @param hidden_dim (int): Hidden Size, the size of hidden states (dimensionality)
        @param candidate_encoder (VocabEmbeddings): VocabEmbeddings object
        @param batch_size (int): batch_size
        """
        super(SentenceBiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embed_dim

        # candidate encoder is an embedding lookup of dimensions embed_dim
        self.candidate_encoder = candidate_encoder
        
        # Initialized the biLSTM
        # input: embedded word representation of dim embed_size
        # output: sentence representation of dim hidden_size
        self.sentence_biLSTM = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.candidate_encoder(sentence).view(len(sentence), self.batch_size, -1)
        encoding, (last_hidden, last_cell)= self.sentence_biLSTM(embeds, self.hidden)

        #`last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards passes.
        # Need to Concatenate the forwards and backwards tensors to obtain the final encoded utterance representation.
        final_utt_rep = torch.cat((last_hidden[0], last_hidden[1]), 1)
        return final_utt_rep

class HierarchicalLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super(HierarchicalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hierarchical_lstm = nn.LSTM(self.input_size, self.hidden_size) 

    def forward(self, x):
        pass

class DialogueActsLSTM(nn.Module):
    def __init__(self):
        super(DialogueActsLSTM, self).__init__()

    def forward(self, x):
        pass

class ClassificationNet(nn.Module):
    """
        Feed-forward network used to determine whether a candidate fills a given slot
    """
    def __init__(self, context_dim, hidden_dim, num_slots):
        """ Init SentenceBiLSTM

        @param context_dim (int): Embedding size (dimensionality)
        @param hidden1_dim (int): Hidden Size, the size of hidden states (dimensionality)
        """
        super(ClassificationNet, self).__init__()
        self.context_feature_dim = context_dim
        self.fc_dim = hidden_dim
        self.num_slots = num_slots

        # FC linear layer: input: context feature vector concatentated with candidate
        #                  outut: vector of dimensions fc_dim
        self.fc1 = nn.Linear(self.context_feature_dim, self.fc_dim)

        # Projection matrix: Projection weight matrix of size (fc_dim x num_slots)
        self.slot_projection = nn.Linear(self.fc_dim, self.num_slots, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output =  self.slot_projection(x)
        return F.logsigmoid(output)



        


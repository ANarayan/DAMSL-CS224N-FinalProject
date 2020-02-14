import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

from model.embeddings import Embeddings
from vocab import Vocab, DAVocab

class DST(nn.Module):
    """ Dialogue State Tracking Model
    """
    def __init__(self, embed_dim, sentence_hidden_dim, hierarchial_hidden_dim, da_hidden_dim, da_embed_size, 
                ff_hidden_dim, batch_size, num_slots, ngrams, candidate_utterance_vocab_pth, da_vocab_pth):
        super(DST, self).__init__()

        # instantiate candidate_embedding
        self.candidate_utterance_vocab = Vocab.load_from_json(candidate_utterance_vocab_pth, ngrams)
        self.da_vocab = DAVocab.load_from_json(da_vocab_pth)
        self.candidate_utterance_embeddings = Embeddings(embed_dim, self.candidate_utterance_vocab)
        self.da_embeddings = Embeddings(da_embed_size, self.da_vocab)
        self.sentence_encoder = SentenceBiLSTM(sentence_hidden_dim, embed_dim, self.candidate_utterance_embeddings, batch_size)
        self.hierarchial_encoder = HierarchicalLSTM(sentence_hidden_dim, hierarchial_hidden_dim)

        self.system_dialogue_acts = DialogueActsLSTM(da_embed_size, da_hidden_dim, batch_size, self.da_embeddings,
                self.da_vocab)
        # context_dim = | [E_i; Z_i; A_i; C_ij] | where E_i = encoded utterance, Z_i = encoding of past user utterances, 
        #                                               A_i = system actions, C_ij = candidate encoding=
        self.context_cand_dim = embed_dim + 2 * (sentence_hidden_dim) + hierarchial_hidden_dim + da_hidden_dim
        self.classification_net = ClassificationNet(self.context_cand_dim, ff_hidden_dim, num_slots)

    def get_turncontext(self, turn):
        """ Compute turn context -- dependent on user utterance, system dialogue acts, 
            and dialogue history
            @param turn (Dict): turn['user_utterance'] : user utterance for the turn (List[String]), 
                            turn['system_actions_formatted'] : agent dialogue acts (List[String]), 
                            turn['utterance_history'] : encoded user utterance history (List[Tensors])
            @return context (Tensor): concated feature vector 
            @return utterance_enc (Tensor): encoded utterance 
        """
        user_utterance = turn['user_utterance']
        system_dialogue_acts = turn['system_actions_formatted']
        past_utterances = turn['utterance_history']
        
        # TODO: Parallelize this computation
        encoded_past_utterances = []
        for utterance in past_utterances:
            #Map string to index tensor
            utt_idx = self.candidate_utterance_vocab.to_idxs_tensor(utterance)
            encoded_sent = self.sentence_encoder(utt_idx)
            encoded_past_utterances.append(encoded_sent)
        
        # get encoded user utterance for the current turn
        utterance_enc = encoded_past_utterances[-1]

        #system_dialogue_acts = List(Strings)
        da_idxs = self.da_vocab.to_idxs_tensor(system_dialogue_acts)
        dialogue_acts_enc = self.system_dialogue_acts(da_idxs)

        #encoded_past_utterances: List[Tensors(Dim: ((sentence_hidden_dim * 2) x 1))]
        dialogue_context_enc = self.hierarchial_encoder(encoded_past_utterances)

        # concatenate three features together to create context featue vector
        context = torch.cat([utterance_enc, dialogue_context_enc, dialogue_acts_enc],1)
        return context


    def forward(self, turn_context, candidate):
        """ Forward pass for a turn/candidate pair
            @param turn_context (Tensor): vector which is a concatenation of encoded utterance
                                    dialogue history, and system actions
            @param candidate (String): slot candidate
            @returns predicted (Tensor): output vector representing the per slot prediction for
                        each candidate (num_slots x 1)
        """
        candidate_idx = self.candidate_utterance_vocab.to_idxs_tensor(candidate)
        embed_cand = self.candidate_utterance_embeddings(candidate_idx)
        feed_forward_input = torch.cat((turn_context, embed_cand))
        output = self.classification_net(feed_forward_input)
        return output
         
class SentenceBiLSTM(nn.Module): 
    """ BiLSTM used to encode individual user utterances 
    """
    def __init__(self, hidden_dim, embed_dim, candidate_encoder, batch_size):
        """ Init SentenceBiLSTM

        @param embed_dim (int): Embedding size (dimensionality)
        @param hidden_dim (int): Hidden Size, the size of hidden states (dimensionality)
        @param candidate_encoder (Embeddings): Embeddings object
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

        if torch.cuda.is_available():
            hidden = (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else: 
            hidden = (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))
        return hidden
   
    def forward(self, sentence_idx):
        embeds = self.candidate_encoder(sentence_idx).view(len(sentence_idx), self.batch_size, -1)
        encoding, (last_hidden, last_cell)= self.sentence_biLSTM(embeds, self.hidden)

        #`last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards passes.
        # Need to Concatenate the forwards and backwards tensors to obtain the final encoded utterance representation.
        final_utt_rep = torch.cat((last_hidden[0], last_hidden[1]), 1)
        return final_utt_rep

class HierarchicalLSTM(nn.Module):
    """
    Encodes sentence 
    """
    def __init__(self, embedding_dim, hidden_dim, batch_size):
        super().__init__()
        self.hierarchical_lstm = nn.LSTM(embedding_dim,hidden_dim)

    def forward(self, encoded_past_utterances):
        """
        --encoded_past_utterances: List[Tensor(embedding_dim * 2, batch_size)] or len src_len

        Returns:
            last_hidden: Tensor(hidden_size, b)
        """
        # NOTE: batch_size=1 currently, may need to think about padding to longest source len if changes
        #stacked_utterances: Tensor(src_len, batch_size, embedding_dim)
        
        stacked_utterances = torch.stack(encoded_past_utterances)
        hidden_sts, (last_hidden, last_cell) = self.hierarchical_lstm(stacked_utterances)

        return last_hidden 

class PreviousStateEncoding(nn.Module):
    def __init__(self, emb_dim, max_n_states):
        """
        Unused in the HyST version of the model
        """
        super().__init__()
        self.emb_size = emb_size
        self.s_dim = max_n_states
        self.emb = nn.Embedding(self.s_dim, self.emb_size)

class DialogueActsLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, batch_size, da_embeddings, vocab):
        super(DialogueActsLSTM, self).__init__()

        """
        --emb_dim: default 50
        --hidden_dim: default 64
        --da_embeddings: Embeddings lookup for dialogue acts
        --vocab: Vocab object
        """
        self.batch_size = batch_size
        self.da_embeddings = da_embeddings
        self.vocab = vocab
        self.da_lstm = nn.LSTM(emb_dim, hidden_dim)

    def forward(self, dialogue_acts_idxs):
        """
        --dialogue_acts_idxs: Tensor(t dialogue acts, max len of dialogue acts in single turn out of last t turns) 

        """
        embs = self.da_embeddings(dialogue_acts_idxs).view(dialogue_acts_idxs.shape[1], batch_size, -1)
        enc_hiddens, (last_hidden, last_cell) = self.da_lstm(embs)
        return last_hidden

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



        


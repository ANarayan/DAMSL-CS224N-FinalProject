import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import random
import numpy as np

from model.embeddings import Embeddings
from vocab import Vocab, DAVocab

class DST(nn.Module):
    """ Dialogue State Tracking Model
    """
    def __init__(self, embed_dim, sentence_hidden_dim, hierarchial_hidden_dim, da_hidden_dim, da_embed_size, 
                ff_hidden_dim, ff_dropout_prob, batch_size, num_slots, ngrams, candidate_utterance_vocab_pth, da_vocab_pth, device):
        super(DST, self).__init__()

        # instantiate candidate_embedding
        self.candidate_utterance_vocab = Vocab.load_from_json(candidate_utterance_vocab_pth, ngrams)
        self.da_vocab = DAVocab.load_from_json(da_vocab_pth, ngrams=None)
        self.candidate_utterance_embeddings = Embeddings(embed_dim, self.candidate_utterance_vocab)
        self.da_embeddings = Embeddings(da_embed_size, self.da_vocab)
        self.sentence_encoder = SentenceBiLSTM(sentence_hidden_dim, embed_dim, self.candidate_utterance_embeddings, batch_size)
        self.hierarchial_encoder = HierarchicalLSTM(sentence_hidden_dim*2, hierarchial_hidden_dim)

        self.system_dialogue_acts = DialogueActsLSTM(da_embed_size, da_hidden_dim, batch_size, self.da_embeddings,
                self.da_vocab)
        # context_dim = | [E_i; Z_i; A_i; C_ij] | where E_i = encoded utterance, Z_i = encoding of past user utterances, 
        #                                               A_i = system actions, C_ij = candidate encoding
        self.context_cand_dim = embed_dim + 2 * (sentence_hidden_dim) + hierarchial_hidden_dim + da_hidden_dim
        self.classification_net = ClassificationNet(self.context_cand_dim, ff_hidden_dim, num_slots, ff_dropout_prob)
        self.device = device

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
        system_dialogue_acts = turn['system_dialogue_acts']
        past_utterances = turn['utterance_history']
        #To optimize with pack_padded_sequence in self.sentence_encoder, if necessary
        utterances_lengths = [len(utt) for utt in past_utterances]
        
        utt_idxs = self.candidate_utterance_vocab.to_idxs_tensor(past_utterances, device=self.device)
        #utt_idxs_packed = pack_padded_sequence(utt_idxs, utterances_lengths)
        encoded_past_utterances = self.sentence_encoder(utt_idxs)
        
        # get encoded user utterance for the current turn
        # self.system_dialogue_acts(da_idxs) returns ((1,512))
        utterance_enc = torch.index_select(encoded_past_utterances, 0, torch.tensor([len(encoded_past_utterances)
            - 1], device=self.device))

        #system_dialogue_acts = List(Strings)
        da_idxs = self.da_vocab.to_idxs_tensor(system_dialogue_acts, isDialogueVocab=True, device=self.device)
        # self.system_dialogue_acts(da_idxs) returns ((1,1,64))
        dialogue_acts_enc = self.system_dialogue_acts(da_idxs).squeeze(dim=0)

        # self.hierarchial_encoder(encoded_past_utterances) returns: ((1, 1, 512))
        dialogue_context_enc = self.hierarchial_encoder(encoded_past_utterances.unsqueeze(1)).squeeze(dim=0)

        # concatenate three features together to create context featue vector
        context = torch.cat([utterance_enc, dialogue_context_enc, dialogue_acts_enc],1)
        return context


    def feed_forward(self, turn_context, candidate):
        """ Forward pass for a turn/candidate pair
            @param turn_context (Tensor): vector which is a concatenation of encoded utterance
                                    dialogue history, and system actions: # Tensor: (batch_size, 1, context_encoding)
            @param candidate (String): slot candidate
            @returns predicted (Tensor): output vector representing the per slot prediction for
                        each candidate (num_slots x 1)
        """
        candidate_idx = self.candidate_utterance_vocab.to_idxs_tensor(candidate).cuda() 
        embed_cand = self.candidate_utterance_embeddings.embeddings(candidate_idx).permute(1,0,2) 
        feed_forward_input = torch.cat((turn_context, embed_cand), dim=2)
        output = self.classification_net(feed_forward_input)
        return output

    def forward(self, turn_and_cand):
        context_vectors = torch.stack([self.get_turncontext(cand_dict) for cand_dict in turn_and_cand])
        candidates = [cand_dict['candidate'] for cand_dict in turn_and_cand]
        output = self.feed_forward(context_vectors, candidates) # Tensor: (batch_size, 1, embed_size)
        return output.squeeze(dim=1)
         
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
        #self.batch_size = batch_size
        self.batch_size = 1
        self.hidden_dim = hidden_dim
        self.embedding_dim = embed_dim

        # candidate encoder is an embedding lookup of dimensions embed_dim
        self.candidate_encoder = candidate_encoder
        
        # Initialized the biLSTM
        # input: embedded word representation of dim embed_size
        # output: sentence representation of dim hidden_size
        self.sentence_biLSTM = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
   
    def forward(self, sentence_idx):
        embeds = self.candidate_encoder.embeddings(sentence_idx).permute(1,0,2).contiguous()
        encoding, (last_hidden, last_cell)= self.sentence_biLSTM(embeds)

        #`last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards passes.
        # Need to Concatenate the forwards and backwards tensors to obtain the final encoded utterance representation.

        final_utt_reps = torch.cat((last_hidden[0], last_hidden[1]), 1)
        return final_utt_reps

class HierarchicalLSTM(nn.Module):
    """
    Encodes sentence 
    """
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.hierarchical_lstm = nn.LSTM(embedding_dim,hidden_dim)

    def forward(self, encoded_past_utterances):
        """
        --encoded_past_utterances: List[Tensor(batch_size, utterance_emb_dim * 2)] or len src_len

        Returns:
            last_hidden: Tensor(1, b, hidden_dim)
        """
        hidden_sts, (last_hidden, last_cell) = self.hierarchical_lstm(encoded_past_utterances)
        return last_hidden 

class PreviousStateEncoding(nn.Module):
    def __init__(self, emb_dim, max_n_states):
        """
        Unused in the HyST version of the model
        """
        super().__init__()
        self.emb_size = emb_dim
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
        self.embed_dim = emb_dim
        self.batch_size = batch_size
        self.da_embeddings = da_embeddings
        self.vocab = vocab
        self.da_lstm = nn.LSTM(emb_dim, hidden_dim)

    def forward(self, dialogue_acts_idxs):
        """
        --dialogue_acts_idxs: Tensor(t dialogue acts, max len of dialogue acts in single turn out of last t turns) 

        """
        embs = self.da_embeddings.embeddings(dialogue_acts_idxs).view(dialogue_acts_idxs.shape[1], 1, self.embed_dim)
        enc_hiddens, (last_hidden, last_cell) = self.da_lstm(embs)
        return last_hidden

class ClassificationNet(nn.Module):
    """
        Feed-forward network used to determine whether a candidate fills a given slot
    """
    def __init__(self, context_dim, hidden_dim, num_slots, dropout_prob):
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
        self.dropout = nn.Dropout(dropout_prob)

        # Projection matrix: Projection weight matrix of size (fc_dim x num_slots)
        self.slot_projection = nn.Linear(self.fc_dim, self.num_slots, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x_dropout = self.dropout(x)
        output =  self.slot_projection(x_dropout)
        return output


def get_slot_predictions(output):
    sigmoid_output = F.sigmoid(output)
    predicted_ouputs = torch.round(sigmoid_output)
    return predicted_ouputs



import unittest as ut
import torch
from model.DSTModel import (DST, SentenceBiLSTM, HierarchicalLSTM, PreviousStateEncoding, DialogueActsLSTM,
ClassificationNet)
from model.embeddings import Embeddings
from vocab import Vocab, DAVocab

def setUpModule():

    word_to_id =  {'this': 0, 'is': 1, 'a': 2, 'test': 3}
    vocab = Vocab(1,word_to_id) 

class TestDSTForward(ut.TestCase):

    def setUp(self):
        super().setUp()
        self.batch_size = 2

        word_to_id =  {'this': 0, 'is': 1, 'a': 2, 'test': 3}
        self.vocab = Vocab(1,word_to_id) 

        self.utt_hidden_size = 256

        self.embeddings = Embeddings(300, self.vocab)
        self.sentence_encoder = SentenceBiLSTM(self.utt_hidden_size, 300, self.embeddings, self.batch_size)
        
        da_to_id = {'attraction-inform(choice)': 0, 'hotel-nooffer(none)' : 1, 'restaurant-request(price)':2}
        
        self.da_embed_size = 50
        self.da_hidden_dim = 64
        self.DAVocab = DAVocab(da_to_id)
        self.da_embeddings = Embeddings(self.da_embed_size, self.DAVocab)
        self.dialogue_acts_encoder = DialogueActsLSTM(self.da_embed_size, self.da_hidden_dim, self.batch_size,
                self.da_embeddings, self.DAVocab)
        
        self.hierarchical_hidden_size = 512
        self.hierarchical_lstm = HierarchicalLSTM(self.utt_hidden_size * 2,
                self.hierarchical_hidden_size)
        
        self.ff_hidden_dim = 256
        self.n_slots = 35
        self.classification_net = ClassificationNet(300 + 2 * self.utt_hidden_size +
                self.hierarchical_hidden_size + self.da_hidden_dim, self.ff_hidden_dim, self.n_slots)

    def test_sentenceBiLSTM_forward_dims(self):
        hidden_size = 256
        seq_len = 20
        input = torch.zeros((self.batch_size, seq_len), dtype=torch.long)
        output = self.sentence_encoder(input)
        self.assertEqual(list(output.shape), [2, 512])


    def test_systemdialogueacts_forward_dims(self):
        n_da_in_turn = 2
        input = torch.zeros((1, n_da_in_turn), dtype=torch.long)
        output = self.dialogue_acts_encoder(input)
        #Outputs last hidden of dialogue acts list
        self.assertEqual(list(output.shape), [1,1,self.da_hidden_dim])


    def test_hierarchical_forward_dims(self):
        n_past_utterances = 6
        input = torch.zeros(6, 1, self.utt_hidden_size * 2)
        output = self.hierarchical_lstm(input)
        self.assertEqual(list(output.shape), [1,1, self.utt_hidden_size * 2])
    

    def test_classification_forward(self):
        input = torch.zeros(1, self.classification_net.context_feature_dim)
        output = self.classification_net(input)
        self.assertEqual(list(output.shape), [1, self.n_slots])


if __name__ == '__main__':
    tclasses_to_run = [TestDSTForward]
    loader = ut.defaultTestLoader
    suites = []
    for tc in tclasses_to_run:
        suites.append(loader.loadTestsFromTestCase(tc))
    aggregated_suite = ut.TestSuite(suites)
    runner = ut.TextTestRunner()
    res = runner.run(aggregated_suite)


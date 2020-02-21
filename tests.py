import unittest as ut
import torch
from DSTModel import DST, SentenceBiLSTM, HierarchicalLSTM, PreviousStateEncoding, DialogueActsLSTM,
ClassificationNet

class TestSentenceBiLSTM(ut.TestCase):

    def setUp(self):
        super().setUp()
        

    def test_dim(self):
        pass

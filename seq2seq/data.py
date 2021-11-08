import spacy
import torch
import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random



class TextDataset():
    def __init__(self):
        self.eng_raw = spacy.load("en_core_web_sm")
        self.ger_raw = spacy.load("de_core_news_sm")

#         self.eng_raw = spacy.load("en")
#         self.ger_raw = spacy.load("de")
        self.english = None
        self.german = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
    
    def tokenize_eng(self, text):
        return [token.text for token in self.eng_raw.tokenizer(text)]

    def tokenize_ger(self, text):
        return [token.text for token in self.ger_raw.tokenizer(text)]

    def data_vocab_init(self):
        self.english = Field(tokenize=self.tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
        self.german = Field(tokenize=self.tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
        train_data, validation_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(self.german, self.english))
        
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.english.build_vocab(train_data, max_size=10000, min_freq=2)
        self.german.build_vocab(train_data, max_size=10000, min_freq=2)
        

        


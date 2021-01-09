import torch.nn as nn
import torch   
import spacy
from torchtext import data 
import pandas as pd
import re
import random
import torch.optim as optim

nlp = spacy.load('en')


class classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden, output, n_layers, 
                 bidirectional, dropout):
        
        super().__init__()          
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)

        self.fc = nn.Linear(hidden * 2, output)
        self.act = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        fc_outputs=self.fc(hidden)
        outputs=self.act(fc_outputs)
        
        return outputs

def load_model(path = "best_model.pt"):
    model.load_state_dict(torch.load(path));
    model.eval();

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          
    length = [len(indexed)]                                    
    tensor = torch.LongTensor(indexed).to(device)              
    tensor = tensor.unsqueeze(1).T                             
    length_tensor = torch.LongTensor(length)                   
    prediction = model(tensor, length_tensor)                   
    return prediction.item()                                   

def preprocess(x):
    x = x.replace('https://www.', '')
    x = x.replace('http://www.', '')
    x = x.replace('http://', '')
    x = x.replace('https://', '')

    x = x.replace('.php', '')
    x = x.replace('.json', '')
    x = x.replace('.html', '')
    x = x.replace('.htm', '')
    x = x.replace('1', '')
    x = x.replace('2', '')
    x = x.replace('3', '')
    x = x.replace('4', '')
    x = x.replace('5', '')
    x = x.replace('6', '')
    x = x.replace('7', '')
    x = x.replace('8', '')
    x = x.replace('9', '')
    x = x.replace('%', '')
    x =  re.split('[/\=\-_?\.&\+]', x) 
    return ' '.join([i for i in x if not i.isdigit() and len(i) > 1])


TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)

nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = 100
n_layers = 2
size_of_vocab = len(TEXT.vocab)
n_hidden = 32
n_output = 1
bidir = True
dropout = 0.30 

model = classifier(size_of_vocab, embedding_dim, n_hidden, n_output, n_layers, 
                   bidir, dropout = dropout)


print(predict(model, "overheid gebouw "))

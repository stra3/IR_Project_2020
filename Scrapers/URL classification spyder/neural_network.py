import torch.nn as nn
import spacy
#deal with tensors
import torch

#handling text data
from torchtext import data
import pandas as pd
import re



class classifier(nn.Module):

    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):

        #Constructor
        super().__init__()

        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        #activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):

        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]

        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)

        return outputs

def load_model(path = "best_model.pt"):
    model.load_state_dict(torch.load(path));
    model.eval();

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction
    return prediction.item()

def pre_process(x):
    x = x.replace('https://www.', '')
    x = x.replace('http://www.', '')
    x = x.replace('http://', '')
    x = x.replace('https://', '')

    x = x.replace('.php', '')
    x = x.replace('.json', '')
    x = x.replace('.html', '')
    x =  re.split('[/\=\-_?\.]', x)
    return ' '.join([i for i in x if not i.isdigit() and len(i) > 1])


TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)

nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

#instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,
                   bidirectional = True, dropout = dropout)

print(predict(model, "overheid gebouw "))

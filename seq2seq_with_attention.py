from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
from tqdm import tqdm
from io import open
import unicodedata
import string
import re
import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.legacy.datasets import Multi30k
# from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

# spacy_ger = spacy.load('de_core_news_sm')
# spacy_eng = spacy.load('en_core_web_sm')

# def tokenizer_ger(text):
#     return [tok.text for tok in spacy_ger.tokenizer(text)]
    

# def tokenizer_eng(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]

# german = Field(tokenize = tokenizer_ger, lower = True, init_token = '<sos>', eos_token = '<eos>')
# english = Field(tokenize = tokenizer_eng, lower = True, init_token = '<sos>', eos_token = '<eos>')
# # train_data, validation_data, test_data = Multi30k(language_pair = ('de', 'en'))
# train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
#                                                     fields = (german, english))
# german.build_vocab(train_data, max_size = 10000, min_freq = 2)
# english.build_vocab(train_data, max_size = 10000, min_freq = 2)

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def label_to_string(labels, index2char):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            # if i.item() == EOS_token:
            #     break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                # if j.item() == EOS_token:
                #     break
                sent += f' {index2char[j.item()]}'
            sents.append(sent)

        return sents

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

from data_stuff import dataset, dataloader
data = dataset(input_lang, output_lang, pairs)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout, batch_first = True)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded_input = self.dropout(self.embedding(x)) # shape: (batch_size, seq_length, embedding_size)
        output, context = self.rnn(embedded_input) # context[0] shape: (n_layers, batch_size, hidden_size)
        return context

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, context):
        x = x.unsqueeze(-1) # x shape: (batch_size) -> (batch_size, 1)
        embedded_input = self.dropout(self.embedding(x)) # shape: (batch_size, 1, embedding_size)
        output, context = self.rnn(embedded_input, context) # output shape: (batch_size, 1, hidden_size)
        output = output.squeeze(1) # output shape: (batch_size, hidden_size)
        pred = self.fc(output) # pred shape: (batch_size, output_size)
        return pred, context

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, src, target, teacher_forcing_ratio = 0.5):
        batch_size = target.shape[0]
        target_len = target.shape[-1]
        # target shape: (batch_size, target_len)
        context = self.encoder(src) # shape: (encoder_layers, batch_size, encoder_hidden_size)
        outputs = torch.zeros((batch_size, target_len, self.vocab_size)) # shape: (batch_size, target_len, vocab_size)
        # token = target[:, 0] # shape: (batch_size)
        token = torch.tensor([SOS_token]*batch_size).to(torch.device('cuda'))
        for i in range(0, target_len):
            pred, context = self.decoder(token, context) # pred shape: (batch_size, vocab_size) 
            outputs[:, i] = pred
            best_guess = pred.argmax(-1) # shape: (batch_size, 1)
            token = target[:, i] if random.random() < teacher_forcing_ratio else best_guess
        return outputs

# class Encoder(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, n_layers, dropout):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = nn.Dropout(dropout)
#         self.embedding = nn.Embedding(input_size, embedding_size)
#         self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional = True, dropout = dropout, batch_first = True)

#         self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
#         self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

#     def forward(self, x):
#         # x shape: (batch_size, seq_length)
#         embedded_input = self.dropout(self.embedding(x)) # shape: (batch_size, seq_length, embedding_size)
#         output, context = self.rnn(embedded_input) # output shape: (batch_size, seq_length, hidden_size*2)

#         hidden = self.fc_hidden(torch.cat((context[0][0:2], context[0][2:4]), dim = 2)) # shape: context[0] (n_layers*2, batch_size, hidden_sizse) -> hidden (n_layers, batch_size, hidden_size*2) -> (n_layers, batch_size, hidden_size)
#         cell = self.fc_cell(torch.cat((context[1][0:2], context[1][2:4]), dim = 2)) # shape: context[1] (n_layers*2, batch_size, hidden_sizse) -> cell (n_layers, batch_size, hidden_size*2) -> (n_layers, batch_size, hidden_size)

#         return output, hidden, cell

# class Decoder(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, dropout):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = nn.Dropout(dropout)
#         self.embedding = nn.Embedding(input_size, embedding_size)
#         self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, n_layers, dropout = dropout, batch_first = True)

#         self.energy = nn.Linear(hidden_size*4, 1)
#         self.softmax = nn.Softmax(dim=1)
#         self.relu = nn.ReLU()

#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x, encoder_output, hidden, cell):
#         # encoder_output shape: (batch_size, seq_length, hidden_size*2)
#         # hidden shape: (n_layers, batch_size, hidden_size)
#         x = x.unsqueeze(-1) # x shape: (batch_size) -> (batch_size, 1)
#         embedded_input = self.dropout(self.embedding(x)) # shape: (batch_size, 1, embedding_size)

#         seq_length = encoder_output.shape[1]
#         hidden_reshaped = torch.cat((hidden[0:1], hidden[1:2]), dim = 2)
#         hidden_reshaped = hidden_reshaped.repeat(seq_length, 1, 1) # shape: (seq_length, batch_size, hidden_size*2)
#         hidden_reshaped = hidden_reshaped.permute(1,0,2) # shape: (batch_size, seq_length, hidden_size*2)
#         energy = self.relu(self.energy(torch.cat((encoder_output, hidden_reshaped), dim = 2))) # shape: (batch_size, seq_length, 1)
#         attention = self.softmax(energy) # shape: (batch_size, seq_length, 1)
#         attention = attention.permute(0,2,1) # shape: (batch_size, 1, seq_length)
#         context_vector = torch.bmm(attention, encoder_output) # shape: (batch_size, 1, hidden_size*2)
#         rnn_input = torch.cat((context_vector, embedded_input), dim = 2) # shape: (batch_size, 1, hidden_size*2 + embedding_size)

#         output, (hidden, cell) = self.rnn(rnn_input, (hidden,cell)) # output shape: (batch_size, 1, hidden_size)
#         output = output.squeeze(1) # output shape: (batch_size, hidden_size)
#         pred = self.fc(output) # pred shape: (batch_size, output_size)
#         return pred, hidden, cell

# class seq2seq(nn.Module):
#     def __init__(self, encoder, decoder, vocab_size):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.vocab_size = vocab_size

#     def forward(self, src, target, teacher_forcing_ratio = 0.5):
#         batch_size = target.shape[0]
#         target_len = target.shape[-1]
#         # target shape: (batch_size, target_len)
#         encoder_output, hidden, cell = self.encoder(src) # shape: (encoder_layers, batch_size, encoder_hidden_size)
#         outputs = torch.zeros((batch_size, target_len, self.vocab_size)) # shape: (batch_size, target_len, vocab_size)
#         # token = target[:, 0] # shape: (batch_size)
#         token = torch.tensor([SOS_token]*batch_size).to(torch.device('cuda'))
#         for i in range(0, target_len):
#             pred, hidden, cell = self.decoder(token, encoder_output, hidden, cell) # pred shape: (batch_size, vocab_size) 
#             outputs[:, i] = pred
#             best_guess = pred.argmax(-1) # shape: (batch_size, 1)
#             token = target[:, i] if random.random() < teacher_forcing_ratio else best_guess
#         return outputs

def main():
    mode = 'test'
    load_model = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 50
    learning_rate = 0.001
    batch_size = 64 if mode != 'test' else 1

    encoder_input_size = input_lang.n_words
    decoder_input_size = output_lang.n_words
    output_size = output_lang.n_words
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    n_layers = 2
    encoder_dropout = 0.1
    decoder_dropout = 0.1

    writer = SummaryWriter(f'runs/loss_plot_no_atn')
    step = 0

    encoder = Encoder(input_size = encoder_input_size, embedding_size = encoder_embedding_size, hidden_size = hidden_size, n_layers = n_layers, dropout = encoder_dropout)
    decoder = Decoder(input_size = decoder_input_size, embedding_size = decoder_embedding_size, hidden_size = hidden_size, output_size = output_size, n_layers = n_layers, dropout = decoder_dropout)
    model = seq2seq(encoder, decoder, output_size)
    print(model)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        states = torch.load('models/no_atn.pth')
        model.load_state_dict(states['model'])
        optimizer.load_state_dict(states['optimizer'])
    model = model.to(device)

    loader = dataloader(data, batch_size = batch_size, shuffle = True)
    if mode == 'train':
        model.train()
        best_loss = 10000
        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            total_loss = 0
            total_n = 0
            for batch in tqdm(loader):
                optimizer.zero_grad()
                input_data, output_data, input_lengths, output_lengths = batch
                input_data = input_data.to(device)
                output_data = output_data.to(device)
                output_data.view(-1)

                pred = model(input_data, output_data)
                pred = pred.view(-1, pred.size(-1))
                pred = pred.to(device)

                loss = criterion(pred, output_data.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
                optimizer.step()
                total_n += output_data.size(0)
                total_loss += loss.item()
            average_loss = total_loss/total_n
            writer.add_scalar('Training Loss', average_loss, global_step = epoch)
            print(average_loss)
            if average_loss < best_loss:
                states = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                best_loss = average_loss
                torch.save(states, 'models/no_atn.pth')
    else:
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_n = 0
            for batch in loader:
                input_data, output_data, input_lengths, output_lengths = batch
                input_data = input_data.to(device)
                output_data = output_data.to(device)

                pred = model(input_data, output_data, teacher_forcing_ratio = 0)
                print(f'{label_to_string(input_data, input_lang.index2word)} - {label_to_string(output_data, output_lang.index2word)} - {label_to_string(pred.max(-1)[1], output_lang.index2word)}')
                print(input_data.size(-1), output_data.size(-1), pred.max(-1)[1].size(-1))
                pred = pred.view(-1, pred.size(-1))
                pred = pred.to(device)

                loss = criterion(pred, output_data.view(-1))
                total_n += output_data.size(0)
                total_loss += loss.item()
            print(total_loss/total_n)

if __name__ == '__main__':
    main()
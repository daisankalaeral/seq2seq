from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
    

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize = tokenizer_ger, lower = True, init_token = '<sos>', eos_token = '<eos>')
english = Field(tokenize = tokenizer_eng, lower = True, init_token = '<sos>', eos_token = '<eos>')
# train_data, validation_data, test_data = Multi30k(language_pair = ('de', 'en'))
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (german, english))
german.build_vocab(train_data, max_size = 10000, min_freq = 2)
english.build_vocab(train_data, max_size = 10000, min_freq = 2)

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
        token = target[:, 0] # shape: (batch_size)
        for i in range(1, target_len):
            pred, context = self.decoder(token, context) # pred shape: (batch_size, vocab_size) 
            outputs[:, i] = pred
            best_guess = pred.argmax(-1) # shape: (batch_size, 1)
            token = target[:, i] if random.random() < teacher_forcing_ratio else best_guess
        return outputs

def main():
    epochs = 20
    learning_rate = 0.001
    batch_size = 2

    encoder_input_size = len(german.vocab)
    decoder_input_size = len(english.vocab)
    output_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    n_layers = 2
    encoder_dropout = 0.5
    decoder_dropout = 0.5

    writer = SummaryWriter(f'runs/loss_plot')
    step = 0

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_within_batch = True,
        sort_key = lambda x: len(x.src),
        device = torch.device('cpu')
    )

    encoder = Encoder(input_size = encoder_input_size, embedding_size = encoder_embedding_size, hidden_size = hidden_size, n_layers = n_layers, dropout = encoder_dropout)
    decoder = Decoder(input_size = decoder_input_size, embedding_size = decoder_embedding_size, hidden_size = hidden_size, output_size = output_size, n_layers = n_layers, dropout = decoder_dropout)
    model = seq2seq(encoder, decoder, output_size)
    print(model)

    pad_idx = english.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    print(len(train_iterator))
    for epoch in range(epochs):
        print(f'Epoch [{epoch+1}/{epochs}]')
        cnt = 0
        for batch_idx, batch in tqdm(enumerate(train_iterator)):
            optimizer.zero_grad()
            input_data = batch.src
            target = batch.trg
            input_data = input_data.permute(1,0).contiguous()
            target = target.permute(1,0).contiguous()

            output = model(input_data, target)
            # output shape: (batch_size, target_len, vocab_size)
            output = output[:,1:].reshape(-1, output.shape[-1])
            loss = criterion(output, target[:,1:].reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()

            writer.add_scalar('Training Loss', loss, global_step = step)
            step+=1
            cnt+=1
            if cnt == 20:
                break

if __name__ == '__main__':
    main()
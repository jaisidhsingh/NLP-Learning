import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torchtext


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_dims, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.recurrent_unit = nn.LSTM(embedding_size, hidden_dims, num_layers, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        #input shape: (sentence_length, batch_size)
        embedding = self.dropout(self.embedding(x))
        #embedding shape: (sentence_length, batch_size, embedding_size)

        encoder_state, (hidden, z_cell) = self.recurrent_unit(embedding)
        #encoder state shape: (sequence_length, batch_size, hidden_dims)

        return hidden, z_cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_dims, num_layers, dropout_prob, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.recurrent_unit = nn.LSTM(embedding_size, hidden_dims, num_layers, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dims, output_size)

    def forward(self, x, hidden, z_cell):
        #input shape: (batch_size) convert to => (1, batch_size)
        x = x.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(x))
        #embedding shape: (1, batch_size, embedding_size)

        outputs, (hidden_state, cell) = self.recurrent_unit(embedding, (hidden, z_cell))
        #outputs shape: (1, batch_size, hidden_dims)
        
        predictions = self.fc(outputs).squeeze(0)
        #unsqueeze so that the shape which is read by the loss function is (batch_size, output_size) so that batch isnt considered as 1

        return predictions, hidden_state, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, VOCAB_SIZE):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = VOCAB_SIZE

    def forward(self, source, target, teacher_force_ratio=0.5):
        BATCH_SIZE = source.shape[1]
        TARGET_LEN = target.shape[0]
        TARGET_VOCAB_SIZE = self.vocab_size

        #what the outputs should look like shape-wise: (target_len, batch_size, target_vocab_size)
        outputs = torch.zeros(TARGET_LEN, BATCH_SIZE, TARGET_VOCAB_SIZE)
        #our mask essentially

        hidden, cell = self.encoder(source)

        #start token: <sos> of the recursive pass of the target
        x = target[0]
        
        for i in range(0, TARGET_LEN):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[i] = output
            #stack the outputs in the placeholder/mask we made for the outputs with zeros

            best_prediction = output.argmax(1)
            x = target[i] if random.random() < teacher_force_ratio else best_prediction

        return outputs

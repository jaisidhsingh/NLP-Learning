import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import *
from data import TextDataset
import torchtext.legacy as ttext
from utils import *
from torchtext.legacy.data import Field, BucketIterator


#DATA LOADING FROM THE DATASET CLASS
dataset = TextDataset()
dataset.data_vocab_init()

#HYPERPARAMETER SETUP
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
LOAD_IN_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)
ENCODER_INPUT_DIMS = len(dataset.german.vocab)
DECODER_INPUT_DIMS = len(dataset.english.vocab)
OUTPUT_SIZE = len(dataset.english.vocab)
ENCODER_EMBEDDING_DIMS = 300
DECODER_EMBEDDING_DIMS = 300
HIDDEN_DIMS = 1024
NUM_LAYERS = 2
ENCODER_DROPOUT_PROB = 0.5
DECODER_DROPOUT_PROB = 0.5


train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
            (dataset.train_data, dataset.validation_data, dataset.test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=DEVICE
        )


#LOADING IN THE MODELS
encoder = Encoder(
            ENCODER_INPUT_DIMS,
            ENCODER_EMBEDDING_DIMS,
            HIDDEN_DIMS,
            NUM_LAYERS,
            ENCODER_DROPOUT_PROB
        )

decoder = Decoder(
            DECODER_INPUT_DIMS,
            DECODER_EMBEDDING_DIMS,
            HIDDEN_DIMS,
            NUM_LAYERS,
            DECODER_DROPOUT_PROB,
            OUTPUT_SIZE
        )

encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)

model = Seq2Seq(encoder, decoder, DECODER_INPUT_DIMS).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
pad_idx = dataset.english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

#TRAINING LOSS

test_line = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen"
running_loss = 0.0
for epoch in range(EPOCHS):

    model.eval()
    translation = translate_sentence(test_line, model, dataset.german, dataset.english, DEVICE)
    print("Translation of test sentence: ", translation)

    model.train()
    for idx, batch in enumerate(train_iterator):
        inputs = batch.src.to(DEVICE)
        targets = batch.trg.to(DEVICE)
        outputs = model(inputs, targets)
        outputs = outputs[1:].reshape(-1, outputs.shape[2]).to(DEVICE)
        targets = targets[1:].reshape(-1)

        model.zero_grad()
        loss = criterion(outputs, targets)
        running_loss+=loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print("Loss: ", running_loss)

model.eval()
checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
save_checkpoint(checkpoint)



        

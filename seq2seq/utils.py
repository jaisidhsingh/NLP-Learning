import numpy as np
import torch
from torchtext.data.metrics import bleu_score
import sys
import spacy


def translate_sentence(sentence, model, german, english, device, max_length=50):
    spacy_ger = spacy.load("de_core_news_sm")

    #check if input type is a string
    if (type(sentence) == str):
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    #check if its an array of strings
    else:
        tokens = [token.lower() for token in sentence]
    
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    word_to_idx = [german.vocab.stoi[token] for token in tokens]
    sentence_vector = torch.LongTensor(word_to_idx).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_vector)

    #one hot encoded vector of our starting token
    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(0, max_length):
        previous = torch.LongTensor([outputs[-1]]).to(device)    

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous, hidden, cell)
            best_prediction = output.argmax(1).item()

        outputs.append(best_prediction)
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated = [english.vocab.itos[idx] for idx in outputs]
    #w dont want the start and end tokens making our translated sentence ugly
    translated = translated[1:-1]
    return translated

def check_bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        source = vars(example)["src"]
        targets = vars(example)["trg"]

        predictions = translate_sentence(source, model, german, english, device)
        targets.append([trg])
        outputs.append(predictions)

    return bleu_score(outputs, targets)

def save_checkpoint(model_state_dict, filename="checkpoint_torch.pth.tar"):
    torch.save(model_state_dict, filename)
    print("Checkpoint saved")

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Loading checkpoint")




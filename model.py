import torch.nn as nn
import torch
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SetfAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.heads_dim = embed_size // heads
        self.values = nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.keys = nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.query = nn.Linear(self.head_dims, self.head_dims, bais=False)
        self.fc = nn.Linear(self.embed_size, self.embed_size, bias=False)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], keys.shape[1], query_shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dims)
        keys = keys.reshape(N, key_len, self.heads, self.head_dims)
        query = query.reshape(N, query_len, self.heads, self.head_dims)
        
        # required energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # -1e20 for numerical stability

        attention = nn.softmax(energy/ (math.sqrt(self.embed_size)), dim=3)

        # required out shape: (N, query_len, heads, head_dims)
        out = torch.einsum("nhql, nlhd-> nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dims)
        final = self.fc(out)
        return final


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion*embed_size), nn.ReLU(), nn.Linear(forward_expansion*embed_size, embed_size))

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        fw = self.feed_forward(x)
        out = self.dropout(self.norm2(fw+x))
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)])
        self.dropout = nn.Embedding(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.positional_embeddings(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transform_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value, key, query, source_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_layers, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, target_vocab_size)

    def forward(self, x, encoder_output, source_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        for layer in self.layer:
            x = layer(encoder_output, encoder_output, source_mask, target_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_pad_index, target_pad_index, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda', max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = source_pad_index
        self.tgt_pad_idx = target_pad_index
        self.dropout = dropout
        self.device = device

    def get_source_mask(self, source):
        source_mask = (source != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # as we require shape (N, 1, 1, source_length)
        return source_mask.to(self.device)

    def grt_target_mask(self, target):
        N, tgt_len = target.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
        return tgt_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.get_source_mask(source)
        tgt_mask = self.get_target_mask(target)
        source_thru_encoder = self.encoder(source, source_mask)
        out = self.decoder(target, source_thru_encoder, source_mask, tgt_mask)
        return out

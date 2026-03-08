import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even idx
        pe[:, 1::2] = torch.cos(position * div_term)  # odd idx
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = F.softmax(scores, dim=-1)
    return attn @ value, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        def split(x):  # (B, T, D) → (B, H, T, d_k)
            return x.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        q = split(self.q_linear(q))
        k = split(self.k_linear(k))
        v = split(self.v_linear(v))

        x, _ = attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        return self.out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x2 = self.norm1(x + self.attn(x, x, x, mask))
        return self.norm2(x2 + self.ff(x2))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.cross_attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x2 = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x3 = self.norm2(x2 + self.cross_attn(x2, enc_out, enc_out, src_mask))
        return self.norm3(x3 + self.ff(x3))

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff) for _ in range(N)])

    def forward(self, x, mask):
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff) for _ in range(N)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=2, heads=8, d_ff=2048):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, d_ff)

    def make_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        no_peek = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask & no_peek
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_mask(src, tgt)
        enc_out = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return output

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import NamedTuple

# https://github.com/dapowan/LIMU-BERT-Public.git
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)

        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        # return self.drop(self.norm(e))
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        # """
        # x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        # * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        # """
        # # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
        #            for x in [q, k, v])
        # # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        # # scores = self.drop(F.softmax(scores, dim=-1))
        # scores = F.softmax(scores, dim=-1) # weights
        # # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        # h = (scores @ v).transpose(1, 2).contiguous()
        # # -merge-> (B, S, D)
        # h = merge_last(h, 2)
        # self.scores = scores
        # return h
        """
        x: (B, S, D)
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.proj_q.out_features // self.n_heads

        # Project and reshape
        q = self.proj_q(x).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = self.proj_k(x).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = self.proj_v(x).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        h = torch.matmul(attn_weights, v)

        # Reshape and concatenate heads
        h = h.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        self.scores = attn_weights
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class PretrainModelConfig(NamedTuple):

    feature_num: int = 0  #
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)
  
# TransformerLayer as defined previously
class TransformerLayer(nn.Module):
    """Single Transformer Layer with Self-Attention and Feed-Forward Networks"""
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
    
    def forward(self, x):
        # Self-Attention
        attn_output = self.attn(x)
        x = self.norm1(x + self.proj(attn_output))
        
        # Feed-Forward Network
        ff_output = self.pwff(x)
        x = self.norm2(x + ff_output)
        
        return x

# Modified Transformer class
class Transformer(nn.Module):
    """Transformer with Independent Self-Attentive Layers"""
    def __init__(self, embed=None, share_layer=True, **kwargs):
        # share_layer == True -> LIMU-Bert.
        super().__init__()
        cfg = PretrainModelConfig(**kwargs)

        if embed is None:
            self.embed = Embeddings(cfg)
        else:
            self.embed = embed

        self.n_layers = cfg.n_layers
        self.share_layer = share_layer
        
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(self.n_layers if not self.share_layer else 1)])
    
    def forward(self, x):
        h = self.embed(x)
        if self.share_layer:
            layer = self.layers[0]
            for _ in range(self.n_layers):
                h = layer(h)
        else:
            for layer in self.layers:
                h = layer(h)

        return h
    
class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = LayerNorm(cfg)
        self.pred = nn.Linear(cfg.hidden, cfg.feature_num)

    def forward(self, input_seqs):
        h_masked = gelu(self.linear(input_seqs))
        h_masked = self.norm(h_masked)
        return self.pred(h_masked)
    
# LIMU-BERT Model
class BertModel4Pretrain(nn.Module):
    def __init__(self, output_embed=False, freeze_encoder=False, freeze_decoder=False, **kwargs):
        super().__init__()
        # Initialize using kwargs that contains necessary parameters in use.
        cfg  = PretrainModelConfig(**kwargs)
        self.encoder = Transformer(**kwargs)

        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.encoder(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        return self.decoder(h_masked)

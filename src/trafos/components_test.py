import torch
import nite
import numpy as np

from trafos.components import attention
from trafos.components import MultiheadAttention
from trafos.components import FeedForwardNetwork
from trafos.components import positional_encoding
from trafos.components import TransformerEncoder


def attention_test():
    keys_dim = 100
    values_dim = 130
    bsize = 64
    length = 200
    keys = torch.rand(bsize, length, keys_dim)
    queries = torch.rand(bsize, length, keys_dim)
    values = torch.rand(bsize, length, values_dim)

    attended = attention(queries, keys, values)
    assert list(attended.shape) == [bsize, length, values_dim]

def multihead_attention_test():
    bsize = 64
    length = 256
    dim = 1024
    mha = MultiheadAttention(8, dim)
    before = torch.rand(bsize, length, dim)
    after = mha(before, before, before)

    assert list(after.shape) == [bsize, length, dim]

def feedforward_network_test():
    embed_dim = 1024
    hidden_dim = 512
    length = 128
    bsize = 64

    ffn = FeedForwardNetwork(embed_dim, hidden_dim)
    before = torch.rand(bsize, length, embed_dim)
    after = ffn(before)
    assert list(after.shape) == [bsize, length, embed_dim]

def positional_encoding_test():
    length = 200
    model_dim = 128
    pe = positional_encoding(length, model_dim)

    assert list(pe.shape) == [length, model_dim]
    y = np.arange(length)
    assert np.all(pe[:,10] == np.sin(y / 10_000 ** (10/model_dim)))
    assert np.all(pe[:,11] == np.cos(y / 10_000 ** (10/model_dim)))

def transformer_encoder_test():
    embed_dim = 256
    attention_heads = 8
    ffn_hidden_dim = 512
    bsize = 64
    length = 200

    encoder = TransformerEncoder(embed_dim, attention_heads, ffn_hidden_dim)
    before = torch.rand(bsize, length, embed_dim)
    after = encoder(before)
    assert before.shape == after.shape

if __name__ == '__main__':
    attention_test()
    multihead_attention_test()
    feedforward_network_test()
    positional_encoding_test()
    transformer_encoder_test()
    print('success')

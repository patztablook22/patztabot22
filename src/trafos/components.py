import torch
import nite
import numpy as np


def attention(queries, keys, values, mask=None):
    assert queries.dim() == keys.dim() == values.dim() == 3
    assert queries.shape[1] == keys.shape[1]
    keydim = keys.shape[1]
    if mask is not None:
        assert mask.dim() == 3 and mask.shape[2] == keydim

    similarities = queries @ keys.mT / np.sqrt(keydim)

    if mask is not None:
        similarities[mask] = -1e7

    return nite.softmax(similarities, dim=-1) @ values

class MultiheadAttention(nite.Module):
    def __init__(self, heads, embed_dim, keys_dim = None, values_dim = None):
        perhead_dim = embed_dim // heads
        if keys_dim is None:
            keys_dim = perhead_dim
        if values_dim is None:
            values_dim = perhead_dim

        self._embed_dim = embed_dim
        self._keys_dim = keys_dim
        self._values_dim = values_dim

        self._heads = [
            [
                torch.nn.Parameter(torch.Tensor(*shape))
                for shape in [(embed_dim, keys_dim),
                              (embed_dim, keys_dim),
                              (embed_dim, values_dim)]
            ]
            for _ in range(heads)
        ]

        self._output = torch.nn.Parameter(
                torch.Tensor(heads * values_dim, embed_dim))

        torch.nn.init.xavier_uniform_(self._output)
        for wq, wk, wv in self._heads:
            torch.nn.init.xavier_uniform_(wq)
            torch.nn.init.xavier_uniform_(wk)
            torch.nn.init.xavier_uniform_(wv)

    def forward(self, queries, keys, values, mask=None):
        assert queries.dim() == keys.dim() == values.dim() == 3
        assert queries.shape == keys.shape == values.shape
        assert queries.shape[2] == self._embed_dim

        if mask is not None:
            assert mask.dim() == 3 and mask.shape[2] == self._embed_dim

        heads = [
            attention(queries @ wq, keys @ wk, values @ wv, mask=mask)
            for wq, wk, wv in self._heads]

        return torch.concat(heads, dim=-1) @ self._output

class FeedForwardNetwork(nite.Module):
    def __init__(self, embed_dim, hidden_dim):
        self._seq = nite.Seq(
            nite.LayerNorm([embed_dim]),
            nite.Dense(embed_dim, hidden_dim, 'gelu'),
            nite.Dense(hidden_dim, embed_dim)
        )

    def forward(self, feed):
        return feed + self._seq(feed)

class TransformerEncoder(nite.Module):
    def __init__(self, 
                 embed_dim, heads, ffn_dim,
                 attn_keys_dim = None, attn_values_dim = None):

        self._multihead = MultiheadAttention(heads,
                                             embed_dim,
                                             keys_dim=attn_keys_dim,
                                             values_dim=attn_values_dim)
        self._multihead_ln = nite.LayerNorm([embed_dim])
        self._ffn = FeedForwardNetwork(embed_dim, ffn_dim)

    def forward(self, feed):
        attended = self._multihead_ln(feed + self._multihead(feed, feed, feed))
        return self._ffn(attended)

def positional_encoding(length, embed_dim):
    x = 10_000 ** (np.arange(0, embed_dim, 2) / embed_dim).reshape([1, -1])
    y = np.arange(length).reshape([-1, 1])
    angles = y / x
    result = np.zeros([length, embed_dim])
    result[:,::2] = np.sin(angles)
    result[:,1::2] = np.cos(angles)
    return result



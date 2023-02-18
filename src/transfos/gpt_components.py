import torch
import nite

from transfos.components import MultiheadAttention, FeedForwardNetwork


class GptDecoder(nite.Module):
    def __init__(self, 
                 embed_dim, heads, ffn_dim, 
                 attn_keys_dim = None, attn_values_dim = None):
        self._multihead = MultiheadAttention(heads,
                                             embed_dim,
                                             keys_dim=attn_keys_dim,
                                             values_dim=attn_values_dim)
        self._multihead_ln = nite.LayerNorm([embed_dim])
        self._ffn = FeedForwardNetwork(embed_dim, ffn_dim)

    def forward(self, feed, mask):
        attended = self._multihead_ln(feed + self._multihead(feed, feed, feed, mask=mask))
        return self._ffn(attended)


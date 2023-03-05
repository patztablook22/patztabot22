import torch
import nite
from trafos.components import positional_encoding
from trafos.gpt_components import GptDecoder

class Gpt1(nite.Module):
    def __init__(self, 
                 vocab_size, 
                 decoders=12,
                 heads=12,
                 embed_dim=768,
                 ffn_dim=4*768,
                 length=512):

        self._embedding = nite.Embedding(vocab_size, embed_dim)

        self._decoders = nite.Seq(*[
            GptDecoder(embed_dim, heads, ffn_dim) for _ in range(decoders)
        ])
            

        self._logits = nite.Dense(embed_dim, vocab_size)

    def forward(self, feed):
        embed = self._embedding(feed)
        embed += positional_encoding(embed.shape[-2], embed.shape[-1])
        return self._logits(self._decoders(embed))

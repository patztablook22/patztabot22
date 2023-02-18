import torch
import nite
from transfos.components import positional_encoding
from transfos.gpt_components import GptDecoder


class Gpt2(nite.Module):
    def __init__(self):
        embed_dim = 1024
        length = 1024
        heads = 8
        ffn_dim = 4 * embed_dim
        vocab_size = 10000

        self._embedding = nite.Embedding(vocab_size, embed_dim)

        decoders = [GptDecoder(embed_dim, heads, ffn_dim) for _ in range(12)]
        self._decoders = nite.Seq(*decoders)

        self._logits = nite.Dense(embed_dim, vocab_size)

    def forward(self, feed):
        embed = self._embedding(feed)
        embed += positional_encoding(embed.shape[-2], embed.shape[-1])
        return self._logits(self._decoders(embed))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import ByteLevelBPETokenizer
import sys


def train_bookcorpus_bpe():
    # tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    # tokenizer.pre_tokenizer = ByteLevel()

    # trainer = BpeTrainer(vocab_size=40000, 
                         # special_tokens=['[UNK]', '[EOF]'])
    tokenizer = ByteLevelBPETokenizer()

    from datasets import load_dataset
    corpus = load_dataset('bookcorpus')
    if not corpus:
        raise RuntimeError

    train = corpus['train']

    def batch_iterator(bsize=1024):
        for begin in range(0, len(train), bsize):
            yield train[begin : begin + bsize]['text']

    def take_first(n, bsize=1024):
        for i, batch in enumerate(batch_iterator(bsize)):
            if i >= n:
                break
            yield batch

    # tokenizer.train_from_iterator(take_first(1000), trainer=trainer)
    tokenizer.train_from_iterator(take_first(1000), vocab_size=40000)

    return tokenizer

def main(argv):
    tokenizer = train_bookcorpus_bpe()
    tokenizer.save(argv[1])

if __name__ == '__main__':
    main(sys.argv)

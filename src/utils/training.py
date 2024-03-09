from torch.utils.data import Dataset
import torch
import numpy as np
import utils.datasets
import dataclasses
from utils.general import *
import copy
from transformers.trainer_callback import TrainerCallback

class MaskedDataset(Dataset):
    def __init__(self, 
                 sequences,
                 max_tokens):

        if isinstance(max_tokens, (int, float)):
            self.max_tokens = lambda x: int(max_tokens)
        else:
            self.max_tokens = max_tokens
        self.examples = self.process_examples(llcat(lmap(self.generate_examples, sequences)))

    def generate_examples(self, sequences: list) -> list:
        if len(sequences) == 0: return []
        keys = list(sequences[0].keys())
        slen = lambda s: len(s['ids'])
        iter_sequences = 0
        iter_sequence = 0
        examples = []

        def make_example():
            nonlocal iter_sequences, iter_sequence
            max_tokens = self.max_tokens(len(examples))
            example = {k: [] for k in keys}
            while (need := max_tokens - slen(example)) > 0 and iter_sequences < len(sequences):
                seq = sequences[iter_sequences]
                avail = slen(seq) - iter_sequence
                take = min(need, avail)
                for k in keys:
                    example[k].extend(list(seq[k][iter_sequence : iter_sequence + take]))
                iter_sequence += take
                if iter_sequence >= slen(seq):
                    iter_sequences += 1
                    iter_sequence = 0
            if len(example[keys[0]]) > 4:
                examples.append(example)

        while iter_sequences < len(sequences):
            make_example()

        return examples

    def process_examples(self, examples: list) -> list:
        keys = list(examples[0].keys())

        def process_example(example):
            example = copy.deepcopy(example)
            for key in keys:
                example[key] = np.array(example[key])
            return example

        return lmap(process_example, examples)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        def make_dict(example):
            inputs = example['ids']
            labels = np.copy(inputs)
            labels[example['mask']] = -100
            return {
                'input_ids': np.array(inputs).squeeze(),
                'attention_mask': np.ones(inputs.shape).squeeze(),
                'labels': np.array(labels).squeeze(),
            }

        example = self.examples[idx]
        if isinstance(example, list):
            return [make_dict(e) for e in example]
        else:
            return make_dict(example)

class DataCollator:
    def __init__(self, tokenizer=None, pad_token_id=None):
        if pad_token_id is None:
            if tokenizer is None:
                raise ValueError("`tokenizer` or `pad_token_id` must be specified")
            if tokenizer.pad_token_id is None:
                raise ValueError("`tokenizer.pad_token_id` is None")
            pad_token_id = tokenizer.pad_token_id

        self.pad_token_id = pad_token_id

    def __call__(self, data): 
        bsize = len(data)
        if bsize == 0: return {}
        slen = lambda s: len(s['input_ids'])
        pad_to = max(map(slen, data))
        keys = ['attention_mask', 'input_ids', 'labels']
        batch = {k: [] for k in keys if k in data[0]}

        for seq in data:
            pad_len = pad_to - slen(seq)

            k = 'attention_mask'
            if k in seq:
                batch[k].append(list(seq[k]) + [0] * pad_len)

            k = 'input_ids'
            if k in seq:
                batch[k].append(list(seq[k]) + [self.pad_token_id] * pad_len)

            k = 'labels'
            if k in seq:
                batch['labels'].append(list(seq[k]) + [-100] * pad_len)

        return {k: torch.tensor(batch[k], dtype=torch.long) for k in batch}

class LogCallback(TrainerCallback):
    BLANK = '‎ '

    def __init__(self):
        import shellbot
        self._shellbot = shellbot
        super().__init__()

    def on_step_end(self, args, state, control, **kwargs):
        cur = state.global_step - 1
        end = state.max_steps
        interval = max(end // 1000, 1)
        if cur % interval == 0 or cur == end - 1:
            self._print_progress(cur, end)

    def on_log(self, args, state, control, logs, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs, flush=True)

    def _print_progress(self, current, total):
        pad = len(str(total))
        self._shellbot.log(LogCallback.BLANK + str(current + 1).rjust(pad), '/', total, ..., overwrite=True)


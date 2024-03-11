#/usr/bin/env python3

from utils.general import *
from utils import datasets, generation, evaluation
import os
import sys
import shellbot
import random

def evaluate(args):
    shellbot.log("Loading pipeline", ...)
    pipeline = get_pipeline(args)
    shellbot.success()

    shellbot.log("Loading data", ...)
    ctxs = get_data(args)
    shellbot.success()

    shellbot.log("Generating...")
    output = []
    for i, ctx in enumerate(ctxs):
        shellbot.log(str(i + 1).ljust(len(str(len(ctxs)))), '/', len(ctxs), ..., overwrite=True)
        hs = [list(pipeline(ctx, continuous=False)) for _ in range(args.n_hypotheses)]
        buff = {}
        buff['hypotheses'] = hs
        if args.outupt_with_ctxs:
            buff['context'] = ctx
        output.append(buff)

    shellbot.success()

    import pickle
    with open(args.output_path, 'wb') as f:
        pickle.dump(output, f)

def get_pipeline(args):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    import torch
    from utils import generation

    debug = False
    if debug:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.add_tokens([tok for tok in datasets.CONTROL_TOKENS.values() if tok != 'eos'])
        actions = [datasets.Action(time=0, user='p', type='message',
                                   data={'body': 'lorem ipsum dolor sit amet'}),
                   datasets.Action(time=0, user='p', type='message',
                                   data={'body': 'another lorem ipsum dolor sit amet'}),
                   datasets.Action(time=0, user='p', type='reaction',
                                   data={'name': '❤️'}),
                   ]
        for a in actions: a.data['followup'] = True
        buff = lmap(datasets.action_to_string, actions)
        untokenized = []
        for _ in range(10):
            sequence = []
            for _ in range(random.randint(1, 4)):
                sequence.append(random.choice(buff))
            untokenized.append(''.join(sequence))

        model = generation.SamplingModel(tokenizer=tokenizer,
                                         untokenized=untokenized)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipeline = generation.Pipeline(model=model, tokenizer=tokenizer, 
                                   agents=['p'],
                                   debug=True)
    return pipeline


def get_data(args):
    import pickle
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    ctxs = data
    return ctxs

def main(args):
    evaluate(args)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog='evaluate')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n-hypotheses', type=int, default=1)
    parser.add_argument('--output-with-contexts', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

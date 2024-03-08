#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets
import shellbot
import sys
import importlib
import asyncio
import random

async def process_message(msg):
    time = msg.created_at.timestamp()
    user = msg.author.name
    body = msg.content
    reactions = []
    for r in msg.reactions:
        async for u in r.users():
            reactions.append(datasets.Reaction(user=u.name, name=r.emoji))

    return datasets.Message(time=time, user=user, body=body, reactions=reactions)

def get_config(args):
    import json
    with open(args.config) as f:
        config = json.load(f)

    token = config.pop('token', None)
    return token, config

def get_model(args):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    import torch
    from utils import generation

    debug = True
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
    stop_token_ids = [198, 628]
    print(f"{stop_token_ids=}", flush=True)

    @genbot.streamer
    def streamer(streams):
        chat = streams[0].data
        importlib.reload(generation)
        pipeline = generation.Pipeline(model=model, tokenizer=tokenizer)
        for action in pipeline(chat):
            streams[0].write(action)

    return streamer


def main(args):
    shellbot.log("Loading config", ...)
    token, config = get_config(args)
    shellbot.success()

    shellbot.log("Loading model", ...)
    model = get_model(args)
    shellbot.success()

    class Patztabot(genbot.Genbot):
        @genbot.gatekeep
        async def attend(self, channel):
            typing = None
            try:
                async with model.stream() as stream:
                    chat = []
                    async for m in (ctx := self.context(channel, limit=16)): 
                        chat.append(await process_message(m))

                    last = ctx._cache[0] if ctx._cache else None
                    async for action in stream(chat[::-1]):
                        type, eoa = action.type, action.data['yeet']['eoa']
                        if type == 'message':
                            if typing is None:
                                typing = await channel.typing().__aenter__()

                            if eoa and (body := action.data['body']):
                                last = await channel.send(body)

                        elif action.type == 'reaction':
                            if eoa and last: await last.add_reaction(action.data['name'])

            except Exception as e:
                shellbot.error(f"Exception caught: {e}")

            if typing is not None: await typing.__aexit__(None, None, None)
            #if not await ctx.current(): await self.attend(channel, force=True)

    shellbot.log("Serving", ...)
    patztabot = Patztabot(**config)
    model.start_process(minimum=1, maximum=1)
    patztabot.run(token)
    shellbot.log("Done")

def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog='run')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

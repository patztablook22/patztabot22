#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets
import shellbot
import sys
import importlib
import asyncio

async def process_message(msg):
    time = msg.created_at.timestamp()
    user = msg.author.name
    body = msg.content
    reactions = []
    for r in msg.reactions:
        async for u in r.users():
            reactions.append(datasets.Reaction(user=u.name, name=r.emoji))

    return datasets.Message(time=time, user=user, body=body, reactions=reactions)

SPECIAL_TOKENS = {'eos': '<|endoftext|>',
                  'types': '<|types|>',
                  'reacts': '<|reacts|>'}

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

    debug = False
    if debug:
        model = generation.FakeModel()
        tokenizer = generation.FakeTokenizer()
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    stop_token_ids = [198, 628]
    print(f"{stop_token_ids=}", flush=True)

    @genbot.streamer
    def streamer(streams):
        stream = streams[0]
        chat = stream.data
        importlib.reload(generation)
        for action in generation.actions(chat, 
                                         model=model, 
                                         tokenizer=tokenizer, 
                                         stop_token_ids=stop_token_ids,
                                         special_tokens=SPECIAL_TOKENS):
            stream.write(action)
        stream.close()

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
                        if action.type == 'message':
                            if action.data.get('types', False) and typing is None:
                                typing = await channel.typing().__aenter__()

                            if body := action.data['body']:
                                last = await channel.send(body)

                        elif action.type == 'reaction':
                            if last: await last.add_reaction(action.data['name'])

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

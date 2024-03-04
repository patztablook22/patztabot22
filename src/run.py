#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets
import shellbot
import sys
import importlib

async def process_message(msg):
    time = msg.created_at.timestamp()
    user = msg.author.name
    body = msg.content
    reactions = []
    for r in msg.reactions:
        async for user in r.users():
            reactions.append(datasets.Reaction(user=user.name, name=r.emoji))

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

    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_eos_id = tokenizer('\n').input_ids[0]

    print(f"{generation_eos_id=}", flush=True)

    def generate(input_ids):
        importlib.reload(generation)
        return generation.generate(input_ids, 
                                   tokenizer=tokenizer,
                                   model=model,
                                   eos_token_id=generation_eos_id)

    @genbot.streamer
    def streamer(streams):
        stream = streams[0]
        try:

            input_ids = tokenizer(stream.data).input_ids
            for _ in range(1):
                print('input_ids', flush=True)
                response_ids = generate(input_ids)
                print('response_ids', flush=True)

                do_break = False
                if tokenizer.eos_token_id in response_ids:
                    eos_pos = response_ids.index(tokenizer.eos_token_id)
                    response_ids = response_ids[:eos_pos - 1]
                    do_break = True

                response = tokenizer.decode(response_ids)
                print('response', flush=True)
                stream.write(response)

                if do_break: break
                input_ids += response_ids

        except Exception as e:
            stream.write(f"Exception: {e}")
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
            try:
                async with model.stream() as stream:
                    chat = []
                    async for m in (ctx := self.context(channel, limit=16)): 
                        chat.append(await process_message(m))

                    actions = datasets.chat_to_actions(chat)
                    prompt = '\n'.join([datasets.action_to_string(a, special_tokens=SPECIAL_TOKENS) for a in actions[::-1]])
                    prompt += '\np: '

                    async for data in stream(prompt):
                        await channel.send(data)
                    print("leaving loop", flush=True)

                if not await ctx.current(): await self.attend(channel, force=True)

            except Exception as e:
                await channel.send(f"Exception caught: {e}")

            print("leaving attend", flush=True)

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

#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets
import shellbot
import sys

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

    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_eos_id = tokenizer('\n').input_ids[0]

    print(f"{generation_eos_id=}", flush=True)

    def generate(input_ids):
        input_ids = input_ids[-500:]
        output_ids = model.generate(input_ids=torch.tensor([input_ids], dtype=torch.long),
                                    max_new_tokens=50,
                                    num_return_sequences=1,
                                    eos_token_id=generation_eos_id,
                                    pad_token_id=tokenizer.pad_token_id)[0]
        return output_ids.tolist()[len(input_ids):]

    @genbot.streamer
    def streamer(streams):
        stream = streams[0]

        input_ids = tokenizer(stream.data).input_ids
        for _ in range(2):
            print('input_ids', flush=True)
            response_ids = generate(input_ids)
            print('response_ids', flush=True)
            response = tokenizer.decode(response_ids)
            print('response', flush=True)
            stream.write(response)
            if tokenizer.eos_token_id in response_ids:
                break

            input_ids += response_ids

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
            async with model.stream() as stream:
                chat = []
                async for m in (ctx := self.context(channel, limit=16)): 
                    chat.append(await process_message(m))

                actions = datasets.chat_to_actions(chat)
                prompt = '\n'.join([datasets.action_to_string(a, special_tokens=SPECIAL_TOKENS) for a in actions[::-1]])
                prompt += '\np: '

                async for data in stream(prompt):
                    await channel.send(data)

            if not await ctx.current(): await self.attend(channel, force=True)

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

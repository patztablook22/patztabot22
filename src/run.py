#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets

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

    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    @genbot.streamer
    def streamer(streams):
        for stream in streams:
            stream.write(f"Echo {len(stream.data)}")
    return streamer


def main(args):
    token, config = get_config(args)
    model = get_model(args)

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
                print(prompt)

                async for data in stream(prompt):
                    await channel.send(data)

            if not await ctx.current(): await self.attend(channel, force=True)

    patztabot = Patztabot(**config)
    model.start_process(minimum=1, maximum=1)
    patztabot.run(token)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog='run')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets

@genbot.streamer
def model(streams):
    for stream in streams:
        stream.write(f"Echo {len(stream.data)}")

class Patztabot(genbot.Genbot):
    @genbot.gatekeep
    async def attend(self, channel):
        async with model.stream() as stream:
            buff = []
            async for msg in (ctx := self.context(channel, limit=16)):
                buff.append(msg)
                buff.append(f"{msg.author.name}: {msg.content}")

            actions = datasets.chat_to_actions(datasets.load_pycord(buff))
            prompt = '\n'.join([datasets.action_to_string(a, special_tokens=special_tokens) for a in actions])

            async for data in stream(buff[-1]):
                await channel.send(data)

        if not await ctx.current(): await self.attend(channel, force=True)

def get_config(args):
    import json
    with open(args.config) as f:
        config = json.load(f)

    token = config.pop('token', None)
    return token, config

def main(args):
    token, config = get_config(args)
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

#!/usr/bin/env python3

import genbot
from utils.general import *
from utils import datasets

@genbot.streamer
def model(streams):
    for stream in streams:
        stream.write(f"Echo {len(stream.data)}")

class Patztabot(genbot.Genbot):

    # We will gatekeep the attend, i.e. only one attend per channel will be running at any one time.
    @genbot.gatekeep
    async def attend(self, channel):
        async with model.stream() as stream:

            buff = []
            async for msg in (ctx := self.context(channel)):
                buff.append(f"{msg.author.name}: {msg.content}")

            for data in stream('\n'.join(buff[::-1])):
                await channel.send(data)

        # If new messages appeared while we were generating ours, more work has to be done.
        if not await ctx.current():
            await self.attend(channel)

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

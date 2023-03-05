import discord
from discord.ext import commands
from discord import app_commands
import sys
import asyncio
from typing import Literal, Optional
from model_init import model_init as model_init_external
import json

class DiscordBot(commands.Bot):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix='/', 
                         intents=intents)

        self.model = None
        self.timed_channels = {}
        self.timed_channels_span = 3 * 60

        self.command_permissions = [(u['name'], str(u['discriminator'])) for u in config['command_permissions']]
        self.chat_permissions = [(u['name'], str(u['discriminator'])) for u in config['chat_permissions']]

    async def setup_hook(self):
        @self.tree.command()
        @app_commands.describe(action='Action type',
                               model='Model name (optional)')
        async def model(interaction: discord.Interaction,
                        action: Literal['init', 'kill'],
                        model: Optional[str]
                        ):
            """Manages underlying language model"""
            await self.model_command(interaction, action, model)

    async def model_init(self, name):
        await self.model_kill()
        try:
            self.model = model_init_external(name)

            print(f'Initialized model {name}')
            activity = discord.Activity(name=name,
                                        details='24/7 in',
                                        type=discord.ActivityType.playing,
                                        assets={'large_image': 'pp11'},
                                        buttons=['Join'])

            await self.change_presence(activity=activity, 
                                       status=discord.Status.online)
        except:
            pass

    def model_active(self):
        return self.model is not None

    async def model_kill(self):
        if self.model is not None:
            self.model.kill()
            self.model = None
            print(f'Killed model')
            await self.change_presence(activity=None, status=None)

    async def model_command(self, interaction, action, model):
        print(self.command_permissions)
        if (interaction.user.name, interaction.user.discriminator) not in self.command_permissions:
            await interaction.response.send_message('permissions not granted')
            return

        if action == 'init':
            await interaction.response.send_message('initializing model...')
            await self.model_init(model)
            if self.model_active():
                await interaction.channel.send('initialized')
                self.pay_attention(interaction.channel, interaction.created_at)
            else:
                await interaction.channel.send('failed')

        elif action == 'kill':
            if not self.model_active():
                await interaction.response.send_message('nothing to kill')
                return

            await interaction.response.send_message('killing model...')
            await self.model_kill()
            await interaction.channel.send('killed')

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        await self.model_init('Gpt1')

    def pay_attention(self, channel, datetime):
        if channel.type in [discord.ChannelType.private,
                            discord.ChannelType.group]:
            return

        self.timed_channels[channel] = datetime

    def should_pay_attention(self, msg):
        if (msg.author.name, msg.author.discriminator) not in self.chat_permissions:
            return False

        if msg.channel.type in [discord.ChannelType.private,
                                discord.ChannelType.group]:
            return True

        if self.user.mention in msg.content.split():
            self.pay_attention(msg.channel, msg.created_at)
            return True
        else:
            if msg.channel not in self.timed_channels:
                return False
            elif (msg.created_at - self.timed_channels[msg.channel]).total_seconds() > self.timed_channels_span:
                del self.timed_channels[msg.channel]
                return False

            self.pay_attention(msg.channel, msg.created_at)
            return True



    async def on_message(self, msg):
        if not self.should_pay_attention(msg):
            return

        print(f'Received a message by {msg.author} {"-" * 64}')
        print(f'...')

        history = [m async for m in msg.channel.history(limit=100)]

        if not self.model_active():
            return
        
        async with msg.channel.typing():
            output = await self.feed(list(reversed(history)))

            if output is None:
                return

            print()
            print('>>>', output)
            await msg.channel.send(output)

    async def feed(self, history):
        processed_history = []
        for m in history:
            author = 'self' if m.author == self.user else m.author.name
            author_processed = '[SELF]' if m.author == self.user else m.author.name

            #print(f'{author}> {m.content}')

            buffer = f'[USER] {author_processed} [CONTENT] {m.content} [EOF]'
            processed_history.append(buffer)

        inputs = '\n'.join(processed_history)

        if self.model is None:
            return

        output = self.model(history[-1].content)
        return output


def main(argv):
    config = json.load(open(argv[1]))
    token = open(argv[2]).read().strip()

    bot = DiscordBot(config)
    bot.run(token)

if __name__ == '__main__':
    main(sys.argv)

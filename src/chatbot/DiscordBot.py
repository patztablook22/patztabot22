import discord
from discord.ext import commands
from discord import app_commands
import sys
import asyncio
from typing import Literal, Optional
import json
import importlib


class DiscordBot(commands.Bot):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix='/', 
                         intents=intents)

        self.model = None
        self.model_name = None
        self.timed_channels = {}
        self.timed_channels_span = 3 * 60

        self.command_permissions = [(u['name'], str(u['discriminator'])) for u in config['command_permissions']]
        self.chat_permissions = [(u['name'], str(u['discriminator'])) for u in config['chat_permissions']]
        self.default_model = config.get('default_model', None)
        self.requires_restart = False

    async def setup_hook(self):
        #self.tree.clear_commands(guild=None)
        #await self.tree.sync()

        @self.tree.command()
        @app_commands.describe(action='Action type',
                               model='Model name (optional)')
        async def model(interaction: discord.Interaction,
                        action: Literal['load', 'unload'],
                        model: Optional[str]):
            """Manages underlying language model"""
            await self.model_command(interaction, action, model)

        @self.tree.command()
        @app_commands.describe(action='Action type')
        async def bot(interaction: discord.Interaction,
                      action: Literal['stop', 'restart', 'status']):
            """Controls the bot service"""
            await self.bot_command(interaction, action)

        await self.tree.sync()

    async def model_load(self, name):
        await self.model_unload()
        print(f'Loading model {name}...')
        try:
            module = importlib.import_module(f'models.{name}')

            self.model = module.load()
            self.model_name = name

            print(f'Loaded model {name}.')
            activity = discord.Activity(name=name,
                                        details='24/7 in',
                                        type=discord.ActivityType.playing,
                                        assets={'large_image': 'pp11'},
                                        buttons=['Join'])

            await self.change_presence(activity=activity, 
                                       status=discord.Status.online)
        except:
            print(f'Loading model {name} failed.')

    def model_active(self):
        return self.model is not None

    async def model_unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            self.model_name = None
            print(f'Unloaded model {self.model_name}.')
            await self.change_presence(activity=None, status=None)

    async def bot_command(self, interaction, action):
        if (interaction.user.name, interaction.user.discriminator) not in self.command_permissions:
            await interaction.response.send_message('permissions not granted')
            return

        if action == 'stop':
            print('Stopping...')
            await interaction.response.send_message('Stopping...')
            await self.close()
            return

        elif action == 'restart':
            print('Restarting...')
            await interaction.response.send_message('Restarting...')
            self.requires_restart = True
            await self.close()
            return

        elif action == 'status':
            print('Dumping bot status...')
            await interaction.response.send_message('*Le status*')
            return

        else:
            interaction.response.send_message('Unknown action.')

    async def model_command(self, interaction, action, model):
        if (interaction.user.name, interaction.user.discriminator) not in self.command_permissions:
            await interaction.response.send_message('permissions not granted')
            return

        if action == 'load':
            await interaction.response.send_message(f'Loading model {model}...')
            await self.model_load(model)
            if self.model_active():
                await interaction.channel.send('Loaded.')
                self.pay_attention(interaction.channel, interaction.created_at)
            else:
                await interaction.channel.send('Failed.')

        elif action == 'unload':
            if not self.model_active():
                await interaction.response.send_message('No model is currently loaded.')
                return

            await interaction.response.send_message('Unloading model {self.model_name}...')
            await self.model_unload()
            await interaction.channel.send('Unloaded.')

        else:
            await interaction.response.send_message("Unknown action.")

    async def on_ready(self):
        print(f'Logged in as {self.user}.')
        if self.default_model is not None:
            await self.model_load(self.default_model)

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
    if bot.requires_restart:
        exit(69)

if __name__ == '__main__':
    main(sys.argv)

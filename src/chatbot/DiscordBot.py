import discord
from discord.ext import commands
from discord import app_commands
import sys
import os
import asyncio
from typing import Literal, Optional
import json
import importlib
from CommandLogger import CommandLogger


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

    async def log(self, message, interaction=None):
        print(message)
        if interaction is not None:
            if interaction.response.is_done():
                interaction.channel.send(message)
            else:
                interaction.response.send_message(message)


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

    async def model_load(self, logger, name):
        print(f'{os.path.dirname(__file__)}/models/{name}.py')
        if not os.path.exists(f'{os.path.dirname(__file__)}/models/{name}.py'):
            await logger.write(f'Unknown model: {name}.')
            return False

        if self.model_loaded():
            await self.model_unload(logger)

        await logger.write(f'Loading model {name}...')

        try:
            module = importlib.import_module(f'models.{name}')

            self.model = module.load()
            self.model_name = name

            activity = discord.Activity(name=name,
                                        details='24/7 in',
                                        type=discord.ActivityType.playing,
                                        assets={'large_image': 'pp11'},
                                        buttons=['Join'])

            await self.change_presence(activity=activity, 
                                       status=discord.Status.online)

            await logger.write(f'Loaded..')
            return True
        except:
            await logger.write(f'Failed.')
            return False

    async def model_unload(self, logger):
        if self.model is None:
            await logger.write('No model is currently loaded.')
            return False

        del self.model
        name = self.model_name
        self.model = None
        self.model_name = None
        await self.change_presence(activity=None, status=None)

        await logger.write(f'Unloaded model {name}.')

        return True

    def model_loaded(self):
        return self.model is not None

    async def bot_restart(self, logger):
        await logger.write('Restarting...')
        del logger.interaction
        self.requires_restart = True
        await self.close()

    async def bot_stop(self, logger):
        await logger.write('Stopping...')
        del logger.interaction
        await self.close()

    async def git_pull(self, logger):
        await logger.write('Running git pull...')
        return_value = os.system(f'cd "{os.path.dirname(__file__)}" && git pull')
        if return_value != 0:
            await logger.write('Failed.')
            return False

        await logger.write('Successfully pulled the repo.')
        return True

    async def bot_update(self, logger):
        if await self.git_pull(logger):
            await self.bot_restart(logger)

    async def bot_status(self, logger):
        await logger.write('*le status or something idk*')

    async def bot_command(self, interaction, action):
        if (interaction.user.name, interaction.user.discriminator) not in self.command_permissions:
            await interaction.response.send_message('Command permissions not granted.')
            return

        logger = CommandLogger(interaction)

        if action == 'stop':
            await self.bot_stop(logger)

        elif action == 'restart':
            await self.bot_restart(logger)

        elif action == 'update':
            await self.bot_update(logger)

        elif action == 'status':
            await self.bot_status(logger)

        else:
            interaction.response.send_message('Unknown action.')

    async def model_command(self, interaction, action, name):
        if (interaction.user.name, interaction.user.discriminator) not in self.command_permissions:
            await interaction.response.send_message('Command permissions not granted.')
            return

        logger = CommandLogger(interaction)

        if action == 'load':
            await self.model_load(logger, name)

        elif action == 'unload':
            await self.model_unload(logger)


    async def on_ready(self):
        print(f'Logged in as {self.user}.')
        if self.default_model is not None:
            await self.model_load(CommandLogger(), self.default_model)

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

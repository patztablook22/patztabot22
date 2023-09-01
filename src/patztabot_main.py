import genbot, discord
import sys, os, time
from configparser import ConfigParser
from finetuned_gpt import FinetunedGpt
from permission_hierarchy import PermissionHierarchy
import numpy as np

def parseIntDict(l):
    buff = {}
    for a in l.split('\n'): 
        a = a.strip().split(' ')
        if len(a) != 2: continue
        buff[int(a[0])] = int(a[1])
    return buff

class Patztabot(genbot.Genbot):
    def __init__(self, config, data_dir):
        super().__init__()
        permission_config = {
            'default': int(config['permissions']['default']),
            'saved': parseIntDict(config['permissions']['saved'])
        }
        self._permissions = PermissionHierarchy(permission_config, 
                                                os.path.join(data_dir, 'permissions.cache'))
        self._data_dir = data_dir
        self._restart = False
        self._last_responded = {}

        @self.slash_command()
        async def shutdown(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("You are not an admin.", ephemeral=True)
                return
            await ctx.respond("Bye!", ephemeral=True)
            await self.close()

        @self.slash_command()
        async def asdf(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("You are not an admin.", ephemeral=True)
                return

            result = ctx.guild
            await ctx.respond(str(result), ephemeral=True)

        @self.slash_command()
        async def ping(ctx):
            await ctx.respond("Pong.", ephemeral=True)

        @self.slash_command()
        async def restart(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("You are not an admin.", ephemeral=True)
                return
            await ctx.respond("Restarting...", ephemeral=True)
            self._restart = True
            await self.close()

        @self.slash_command()
        async def test(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("You are not an admin.", ephemeral=True)
                return
            test_channels = self.test_channels()
            await ctx.respond(f"Found {len(test_channels)} test channels. Starting...", ephemeral=True)
            for channel in test_channels:
                prev = [msg async for msg in channel.history() if msg.author == self.user]
                await channel.delete_messages(prev)
            for channel in test_channels:
                await self.attend(channel)

        @self.slash_command()
        async def generate(ctx, prompt: str):
            if not self._permissions.mod(ctx.author.id):
                await ctx.respond("You are not a mod.", ephemeral=True)
                return

            await ctx.defer(ephemeral=True)
            first = True
            async for output in self.enqueue(lambda: prompt):
                if first: output = f"[{prompt}]{output}"
                first = False
                await ctx.respond(output, ephemeral=True)

        @self.slash_command()
        async def thread(ctx, name: str):
            if not self.is_visible_user(ctx.author, ctx.guild):
                await ctx.respond("You are not visible.", ephemeral=True)
                return

            if isinstance(ctx.channel, discord.Thread):
                await ctx.respond("Can't create a thread here.", ephemeral=True)
                return

            interaction = await ctx.respond("Creating thread...")
            msg = await interaction.original_response()
            await msg.create_thread(name=name)

        @self.slash_command()
        async def reset(ctx):
            if not self.is_visible_user(ctx.author, ctx.guild):
                await ctx.respond("You are not visible.", ephemeral=True)
                return
            await ctx.respond("Done.")

        permissions = self.create_group(name='permissions')

        @permissions.command(name='get')
        async def permGet(ctx, user: discord.User):
            sl = self._permissions.get_saved_level(user.id)
            if sl == -1: await ctx.respond(f"Default ({self._permissions.levels[self._permissions.default]}).", ephemeral=True)
            else: await ctx.respond(self._permissions.levels[sl].capitalize() + ".", ephemeral=True)

        async def get_possible_levels(ctx: discord.AutocompleteContext):
            if 'user' not in ctx.options: return []
            target_id = int(ctx.options['user'])
            author_id = ctx.interaction.user.id
            target_level = self._permissions.get_level(target_id)
            author_level = self._permissions.get_level(author_id)
            if target_level >= author_level: return []
            buff = self._permissions.levels[:author_level]
            if self._permissions.default < author_level: buff.append('default')
            return buff

        @permissions.command(name='set')
        async def permSet(ctx, 
                          user: discord.User, 
                          level: discord.Option(str, autocomplete=discord.utils.basic_autocomplete(get_possible_levels)),
                          ):
            author_level = self._permissions.get_level(ctx.author.id)
            target_level = self._permissions.get_level(user.id)
            if target_level >= author_level:
                    await ctx.respond("Permission not granted.", ephemeral=True)
                    return
            if level in self._permissions.levels:
                l = self._permissions.levels.index(level)
                if l >= author_level:
                    await ctx.respond("Permission not granted.", ephemeral=True)
                    return
                self._permissions.set(user.id, l)
            elif level == 'default':
                if self._permissions.default >= author_level:
                    await ctx.respond("Permission not granted.", ephemeral=True)
                    return
                self._permissions.unset(user.id)
            else:
                await ctx.respond("Unknown permission level.", ephemeral=True)
                return
            await ctx.respond("Permission level set successfully.", ephemeral=True)

    def text_channel_prefix(self, channel, prefix):
        if not isinstance(channel, discord.TextChannel): return False
        if channel.name.startswith(prefix): return True
        return channel.category and channel.category.name.startswith(prefix)

    def test_channels(self):
        buff = []
        for guild in self.guilds:
            for channel in guild.channels:
                if self.text_channel_prefix(channel, "patztabot22-test"):
                    buff.append(channel)
        return buff


    def worker(self):
        response_limit = 2
        model_name = os.path.join(self._data_dir, "chat_model9")

        #gpt = FinetunedGpt(model_name)
        while True:
            handler = self.consume(max_size=1)[0]
            data = handler.get_data()
            if not data:
                handler.close()
                continue

            handler.wait()
            handler.write('pong')
            handler.close()
            continue
            for _ in range(response_limit):
                out = gpt.predict([data])[0].strip().split('[BREAK]')[0]
                if '[MEND]' in out:
                    out = out.split('[MEND]')[0]
                    if out: handler.write(out)
                    break
                if out: handler.write(out)
                data += out

            handler.close()


    def is_visible_message(self, message):
        MT = discord.MessageType
        if message.type not in [MT.default]: return False
        if len(message.content) < 2: return False
        return self.is_visible_user(message.author, message.guild)

    def is_visible_user(self, member, guild):
        if isinstance(member, (discord.User, 
                               discord.abc.User, 
                               discord.user.User)): return True
        role = 'visible' in map(lambda r: r.name, member.roles)
        if role and "patztabot22" in str(guild): return True
        return False

    async def on_message(self, message):
        if message.author == self.user: return
        if not self.is_visible_message(message): return
        channel = message.channel

        trigger = False
        if isinstance(channel, discord.DMChannel):
            trigger = True
        elif self.user in message.mentions:
            trigger = True
        elif self.text_channel_prefix(channel, "patztabot22-direct"):
            trigger = True
        elif isinstance(channel, discord.Thread) and channel.owner == self.user:
            trigger = True

        if not trigger: return
        await self.attend(message.channel)

    async def attend(self, channel):

        def is_reset_confirmation(msg):
            if msg.author != self.user: return False
            if not hasattr(msg, 'interaction') or not msg.interaction: return False
            appcmd = 2 # discord.InteractionType.application_command
            if msg.interaction.type != appcmd: return False
            return msg.interaction.name == 'reset'

        def is_hidden(msg):
            return msg.content.startswith("!")

        async def make_prompt():
            content = []
            l = 0
            last_trigger = False
            async for m in channel.history(limit=1000, oldest_first=False): 
                if is_reset_confirmation(m): break
                if not self.is_visible_message(m): continue
                if is_hidden(m): continue
                if not last_trigger and m.author != self.user:
                    if self._last_responded.get(channel, -1) >= m.id: return
                    self._last_responded[channel] = m.id
                    last_trigger = True

                u = 'you' if m.author == self.user else m.author.name
                c = m.content
                buff = f'[MSTART]{u}[WRITES]{c}[MEND][BREAK]\n'
                l += len(buff)
                if l > 500: break
                content.append(buff)
                if len(content) > 8: break

            if len(content) == 0: return None
            content = content[::-1]
            content.append('[MSTART]you[WRITES]')
            return ''.join(content)


        turn = await self.enqueue(make_prompt)
        if turn.closed: return
        async with channel.typing():
            async for output in turn:
                await channel.send(output)

def main(argv):
    config = ConfigParser()
    config.read(argv[1])
    token = open(argv[2]).read().strip()
    data_dir = argv[3]
    bot = Patztabot(config, data_dir)
    bot.run(token)

    # the outer shell loops this python script as long as it returns 69
    if bot._restart: exit(69)

if __name__ == '__main__':
    main(sys.argv)

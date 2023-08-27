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

        @self.slash_command()
        async def shutdown(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("Permission not granted.")
                return
            await ctx.respond("Bye!")
            await self.close()

        @self.slash_command()
        async def ping(ctx):
            await ctx.respond("Pong.")

        @self.slash_command()
        async def restart(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("Permission not granted.")
                return
            await ctx.respond("Restarting...")
            self._restart = True
            await self.close()

        @self.slash_command()
        async def test(ctx):
            if not self._permissions.admin(ctx.author.id):
                await ctx.respond("Permission not granted.")
                return
            test_channels = self.test_channels()
            await ctx.respond(f"Found {len(test_channels)} test channels. Starting...")
            for channel in test_channels:
                prev = [msg async for msg in channel.history() if msg.author == self.user]
                await channel.delete_messages(prev)
            for channel in test_channels:
                await self.attend(channel)

        permissions = self.create_group(name='permissions')


        @permissions.command(name='get')
        async def permGet(ctx, user: discord.User):
            sl = self._permissions.get_saved_level(user.id)
            if sl == -1: await ctx.respond(f"Default ({self._permissions.levels[self._permissions.default]}).")
            else: await ctx.respond(self._permissions.levels[sl].capitalize() + ".")

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
                          level: discord.Option(str, autocomplete=discord.utils.basic_autocomplete(get_possible_levels))):
            author_level = self._permissions.get_level(ctx.author.id)
            target_level = self._permissions.get_level(user.id)
            if target_level >= author_level:
                    await ctx.respond("Permission not granted.")
                    return
            if level in self._permissions.levels:
                l = self._permissions.levels.index(level)
                if l >= author_level:
                    await ctx.respond("Permission not granted.")
                    return
                self._permissions.set(user.id, l)
            elif level == 'default':
                if self._permissions.default >= author_level:
                    await ctx.respond("Permission not granted.")
                    return
                self._permissions.unset(user.id)
            else:
                await ctx.respond("Unknown permission level.")
                return
            await ctx.respond("Permission level set successfully.")

    def test_channels(self):
        buff = []
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name.startswith("patztabot22-test-"):
                    buff.append(channel)
        return buff

    def worker(self):
        gpt = FinetunedGpt(os.path.join(self._data_dir, "chat_model5"))
        while True:
            handler = self.consume(max_size=1)[0]
            data = handler.get_data()
            if not data:
                handler.close()
                continue
            
            # first response
            out = gpt.predict([data])[0]
            handler.write(out)

            # try followups
            if len(out) < 20 and np.random.random() < 0.8:
                data = data + out + "[MEND]\n[MSTART]patz[WRITES]"
                out = gpt.predict([data])[0]
                handler.write(out)

            handler.close()

    async def on_message(self, message):
        if not self._permissions.chat(message.author.id):
            return
        if isinstance(message.channel, discord.DMChannel) \
                or self.user in message.mentions:
            await self.attend(message.channel)

    async def attend(self, channel):
        async def make_prompt():
            content = []
            l = 0
            async for m in channel.history(limit=1000, oldest_first=False): 
                u = m.author.name
                c = m.content
                if c == '_break_': break
                if c.startswith("_skip_"): continue
                buff = f'[MSTART]{u}[WRITES]{c}[MEND]\n'.replace(
                        self.user.name, 'patz')
                l += len(buff)
                if l > 500: break
                content.append(buff)
                if len(content) > 8: break

            if len(content) == 0: return None
            content = content[::-1]
            content.append('[MSTART]patz[WRITES]')
            return ''.join(content)

        async with channel.typing():
            async for output in self.enqueue(make_prompt):
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

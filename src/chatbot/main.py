import genbot
import sys, os, time
from configparser import ConfigParser
from finetuned_gpt import FinetunedGpt

def parseList(l):
    buff = []
    for a in l.split('\n'): 
        a = a.strip()
        if a: buff.append(a)
    return buff

class Patztabot(genbot.Genbot):
    def __init__(self, config, data_dir, log):
        super().__init__()
        self._chat_whitelist = list(map(int, parseList(config['Chat']['whitelist'])))
        self._jobs_whitelist = list(map(int, parseList(config['Jobs']['whitelist'])))
        self._data_dir = data_dir
        self._log = log
        self._restart = False

        @self.slash_command()
        async def shutdown(ctx):
            await ctx.respond("Bye!")
            await self.close()

        @self.slash_command()
        async def ping(ctx):
            await ctx.respond("Pong.")

        @self.slash_command()
        async def restart(ctx):
            await ctx.respond("Restarting...")
            self._restart = True
            await self.close()

        @self.slash_command()
        async def test(ctx):
            test_channels = self.test_channels()
            await ctx.respond(f"Found {len(test_channels)} test channels. Starting...")
            for channel in test_channels:
                prev = [msg async for msg in channel.history() if msg.author == self.user]
                await channel.delete_messages(prev)
            for channel in test_channels:
                await self.attend(channel)

    def test_channels(self):
        buff = []
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name.startswith("patztabot22-test-"):
                    buff.append(channel)
        return buff

    def worker(self):
        print("loading gpt... ", end="", file=self._log)
        gpt = FinetunedGpt(os.path.join(self._data_dir, "chat_model3"))
        print("done", file=self._log)
        print("starting worker loop", file=self._log)
        while True:
            handler = self.consume(max_size=1)[0]
            data = handler.get_data()
            if not data:
                handler.close()
                continue
            
            print(data, file=self._log)
            print("generating response> ", end="", file=self._log)
            out = gpt.predict([data])[0]
            print(out, file=self._log)
            handler.write(out)
            handler.close()

    async def on_message(self, message):
        if message.author.id in self._chat_whitelist:
            await self.attend(message.channel)

    async def attend(self, channel):
        async def make_prompt():
            content = []
            l = 0
            async for m in channel.history(limit=1000, oldest_first=False): 
                u = m.author.name
                u = "patz" if u == "patztabot22" else u
                c = m.content
                if c == '_break_': break
                if c.startswith("_skip_"): continue
                buff = f'[MSTART]{u}[WRITES]{c}[MEND]\n'
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
    with open(os.path.join(data_dir, "discord_bot_log.txt"), 'w') as log:
        bot = Patztabot(config, data_dir, log)
        bot.run(token)
        if bot._restart: exit(69)

if __name__ == '__main__':
    main(sys.argv)

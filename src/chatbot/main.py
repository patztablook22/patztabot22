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
    def __init__(self, config, data_dir):
        super().__init__()
        self._chat_whitelist = list(map(int, parseList(config['Chat']['whitelist'])))
        self._jobs_whitelist = list(map(int, parseList(config['Jobs']['whitelist'])))
        self._data_dir = data_dir

        @self.slash_command()
        async def shutdown(ctx):
            await ctx.respond("Bye!")
            await self.close()

        @self.slash_command()
        async def ping(ctx):
            await ctx.respond("Pong.")

    def worker(self):
        #gpt = FinetunedGpt(os.path.join(self._data_dir, "chat_model"))
        while True:
            handler = self.consume(max_size=1)[0]
            data = handler.get_data()
            if not data:
                handler.close()
                continue

            #out = gpt.predict([data])[0]
            #out = data
            #out = out[len(data):]
            time.sleep(1)
            handler.write("nya-")
            handler.close()

    async def on_message(self, message):
        if message.author.id not in self._chat_whitelist:
            return

        async def make_prompt():
            channel = message.channel
            content = []
            async for m in channel.history(limit=3, oldest_first=False): 
                u = message.author.name
                u = "patz" if u == "patztabot22" else u
                c = m.content
                if c == '_break_': break
                content.append(f'[MSTART]{u}[WRITES]{c}[MEND]')

            content = content[::-1]
            content.append('[MSTRART]patz[WRITES]')
            return '\n\n'.join(content)

        async with message.channel.typing():
            async for output in self.enqueue(make_prompt):
                await message.channel.send(output)

def main(argv):
    config = ConfigParser()
    config.read(argv[1])
    token = open(argv[2]).read().strip()
    data_dir = argv[3]
    bot = Patztabot(config, data_dir)

    bot.run(token)

if __name__ == '__main__':
    main(sys.argv)

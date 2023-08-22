import genbot
import sys
import time
from configparser import ConfigParser
from gpt2 import Gpt2

def parseList(l):
    buff = []
    for a in l.split('\n'): 
        a = a.strip()
        if a: buff.append(a)
    return buff

class Patztabot(genbot.Genbot):
    def __init__(self, config):
        super().__init__()
        self._chat_whitelist = list(map(int, parseList(config['Chat']['whitelist'])))
        self._jobs_whitelist = list(map(int, parseList(config['Jobs']['whitelist'])))

    def worker(self):
        gpt = Gpt2()
        while True:
            handler = self.consume(max_size=1)[0]
            data = handler.get_data()
            if not data:
                handler.close()
                continue

            out = gpt.predict([data])[0]
            #out = out[len(data):]
            handler.write(out)
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

        async for output in self.enqueue(make_prompt):
            await message.channel.send(output)

def main(argv):
    config = ConfigParser()
    config.read(argv[1])
    token = open(argv[2]).read().strip()
    bot = Patztabot(config)

    bot.run(token)

if __name__ == '__main__':
    main(sys.argv)

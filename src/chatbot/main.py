import genbot
import sys
import models.GPT1 as GPT1
import jobs
import json

def main(argv):
    config = json.load(open(argv[1]))
    token = open(argv[2]).read().strip()
    bot = genbot.Genbot(models=[GPT1], jobs=jobs, **config)
    bot.run(token)

if __name__ == '__main__':
    main(sys.argv)

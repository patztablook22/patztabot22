import genbot
import sys
import json
import models
import jobs

def main(argv):
    config = json.load(open(argv[1]))
    token = open(argv[2]).read().strip()
    bot = genbot.Genbot(models=models, jobs=jobs, **config)
    bot.run(token)

if __name__ == '__main__':
    main(sys.argv)

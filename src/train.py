import shellbot, discord
import sys, os
from configparser import ConfigParser

def main(argv): 
    config = ConfigParser()
    config.read(argv[1])
    token = open(argv[2]).read().strip()
    data_dir = argv[3]
    shell = shellbot.Shellbot()

    shell.run(token)


if __name__ == '__main__':
    main(sys.argv)

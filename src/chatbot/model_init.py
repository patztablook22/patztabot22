import sys
import importlib

def model_init(name):
    module = importlib.import_module(f'{name}')
    return module.init()

def main(argv):
    model_init(argv[1])

if __name__ == '__main__':
    main(sys.argv)

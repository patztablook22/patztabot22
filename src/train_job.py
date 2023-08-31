import time
import shellbot

i = 0
while True:
    time.sleep(1)
    shellbot.log('ping', i)
    i += 1

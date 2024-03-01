import datetime

def _timed_yielder(f, l):
    t0 = datetime.datetime.now()
    buff = []
    eta0 = None
    eta = None
    last_eta = None
    for i, x in enumerate(l):
        for y in f(x):
            buff.append(y)

        t = datetime.datetime.now()
        eta = (t - t0).total_seconds() / (i + 1) * (len(l) - i - 1)
        if eta0 is None:
            if (i > 10 or i > len(l) / 10) and eta > 5:
                eta0 = eta

        if eta0 is not None:
            if (last_eta is None or last_eta - eta >= last_eta * 0.5) and eta > 10:
                print(f"eta {eta:.0f} s")
                last_eta = eta

    return buff

def lmap(f, l):
    def ff(x): yield f(x)
    return _timed_yielder(ff, l)

def lfilter(p, l):
    def pp(x): 
        if p(x): yield x
    return _timed_yielder(pp, l)

def llmap(f, ll):
    def ff(l): yield list(map(f, l))
    return _timed_yielder(ff, ll)

def llfilter(p, ll):
    def pp(l): 
        if p(l): yield list(filter(p, l))
    return _timed_yielder(pp, ll)

def llcat(ll):
    buff = []
    for l in ll: buff += l
    return buff

import dataclasses
import json
from datetime import datetime
import os
import numpy as np
import langid
import collections
from typing import Optional, Callable
import re
import time
import copy

try:
    import IPython.display
except:
    IPython = None


@dataclasses.dataclass
class Reaction:
    user: str
    name: str

@dataclasses.dataclass
class Issue:
    span: tuple[int, int]
    type: str
    data: dict = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class Message:
    time: float
    user: str
    body: str
    attachments: list[str] = dataclasses.field(default_factory=list)
    reactions: list[Reaction] = dataclasses.field(default_factory=list)
    issues: list[Issue] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Action:
    time: float
    user: str
    type: str
    data: dict = dataclasses.field(default_factory=dict)

def load_cached(path: str) -> list[Message]:
    data = json.load(open(path))
    buff = []
    for d in data['messages']:
        m = Message(**d)
        buff.append(m)
    return buff

def save_cached(chat: list[Message], path: str):
    buff = [dataclasses.asdict(m) for m in chat]
    json.dump({'messages': buff}, open(path, 'w'))

def load_discord(path: str) -> list[Message]:
    data = json.load(open(path))
    messages = []
    for d in data['messages']:
        time = datetime.fromisoformat(d['timestamp']).timestamp()
        user = d['author']['name']
        body = d['content']

        assert isinstance(time, float) 
        assert isinstance(user, str)
        assert isinstance(body, str)

        attachments = [a['fileName'] for a in d['attachments']]
        for a in attachments: assert isinstance(a, str)

        reactions = []
        for r in d['reactions']:
            for u in r['users']:
                assert isinstance(u['name'], str)
                assert isinstance(r['emoji']['name'], str)
                reactions.append(Reaction(user=u['name'], name=r['emoji']['name']))

        messages.append(Message(time=time, user=user, body=body,
                                attachments=attachments, reactions=reactions))

    return messages

def load_messenger(path: str) -> list[Message]:
    if os.path.isdir(path):
        root = path
        jsons = list(filter(lambda x: x.endswith('json'), os.listdir(root)))
    else:
        root = os.path.dirname(path)
        jsons = [os.path.basename(path)]

    def unfuck(s):
        return s.encode('latin1').decode('utf8')
    
    attachment_keys = ['photos', 'videos', 'audio_files', 'files']

    messages = []
    for j in jsons:
        with open(os.path.join(root, j), 'r') as f:
            data = json.load(f)

        for d in data['messages']:
            user = unfuck(d['sender_name'])
            time = d['timestamp_ms'] / 1000
            if 'content' in d:
                body = unfuck(d['content'])
            else:
                body = ""

            assert isinstance(user, str)
            assert isinstance(time, float)
            assert isinstance(body, str)

            attachments = []
            for key in attachment_keys:
                if key not in d: continue
                for a in d[key]:
                    attachments.append(os.path.basename(a['uri']))

            reactions = []
            if 'reactions' in d:
                for r in d['reactions']:
                    reactions.append(Reaction(user=unfuck(r['actor']),
                                              name=unfuck(r['reaction'])))


            messages.append(Message(time=time, user=user, body=body,
                                    attachments=attachments, reactions=reactions))

    return sorted(messages, key=lambda m: m.time)

def get_users_list(chat: list) -> list[str]:
    users = set()
    for m in chat:
        users.add(m.user)
        for r in m.reactions: users.add(r.user)

    return list(users)

def make_chat_view(chat, indices):
    if not IPython: raise RuntimeError("IPython is missing")

    def message_view(m: Message):
        user = m.user
        issues_by_pos = [(i.span[0], i) for i in m.issues] + \
                        [(i.span[1], i) for i in m.issues]
        issues_by_pos = sorted(issues_by_pos, key=lambda i: -i[0])
        content = m.body
        for p, i in issues_by_pos:
            if p == i.span[0]:
                color = {'censor': 'rgba(255, 0, 0, 0.5)'}.get(i.type, 'purple')
                tag = f"""<span style="background: {color}">"""
            else:
                tag = """</span>"""

            content = content[:p] + tag + content[p:]
            

        content += " "
        for a in m.attachments:
            content += f"[{a}]"

        reactions = ""
        if m.reactions:
            for r in m.reactions:
                reactions += f"""<li style="display: inline">{r.name}</li>"""
            reactions = f"""<ul style="margin: 10px 0px 0px -13px; padding: 0px 0px 0px 10px; border-left: 3px solid grey">
                            {reactions}
                        </ul>"""

        return f"""<tr>
                    <td style="vertical-align: top; white-space: nowrap; width: 10em; overflow: hidden; text-overflow: ellipsis">{user}</td>
                    <td style="text-align: left;">{content}{reactions}</td></tr>"""

    buff = ''.join([message_view(chat[i]) for i in indices])
    
    IPython.display.display(IPython.display.HTML(f"""<table style="width: 100%">{buff}</table>"""))

def view_head(chat, n=10):
    make_chat_view(chat, range(n))

def view_tail(chat, n=10):
    n = min(n, len(chat))
    make_chat_view(chat, range(-n, 0))

def view_excerpt(chat, n = 10, where=None):
    if where:
        candidates = [max(0, min(i, len(chat) - n)) for i, m in enumerate(chat) if where(m)]
        if not candidates:
            print("No such excerpt found.")
            return
        begin = np.random.choice(candidates) - 1
    else:
        if len(chat) <= n:
            begin = 0
        else:
            begin = np.random.choice(max(1, len(chat) - n))

    make_chat_view(chat, range(begin, min(len(chat) - 1, begin + n)))

def get_languages(ms, languages=None, threshold=0.05):
    if languages: langid.set_languages(languages)
    batch = True
    if isinstance(ms, Message):
        batch = False
        ms = [ms]

    def softmax(x, **kwargs):
        ex = np.exp(x)
        return ex/(ex.sum(**kwargs) + 1e-7)

    buff = []
    classified = 0
    for m in ms:
        body = m.body
        if len(body) >= 3:
            # ls, confs = zip(*langid.rank(body))
            # confs = softmax(confs)
            # i = np.argmax(confs)
            # l, conf = ls[i], confs[i]
            # if conf > 0.90:
                # classified += 1
            # else:
                # l = None
            classified += 1
            l, conf = langid.classify(body)
        else:
            l = None
        buff.append(l)
    
    if not batch:
        return buff[0]

    counter = collections.Counter(buff)
    buff1 = {}
    for l in counter:
        if not l: continue
        f = counter[l] / classified
        if f < threshold: continue
        buff1[l] = f
    return buff1

def get_users(ms: list[Message], threshold=0.05):
    buff = []
    total = 0
    for m in ms:
        if not m.user: continue
        total += 1
        buff.append(m.user)

    counter = collections.Counter(buff)
    buff = {}
    for user in counter:
        if not user: continue
        f = counter[user] / total
        if f < threshold: continue
        buff[user] = f
    return buff

# def plot_summary(ms, ax):
    # ax.axis('off')
    # text = f"Total messages: {len(ms)}"
    # ax.text(0, 0.5, text, fontsize=12, wrap=True)

# def plot_languages(ms, ax, languages=None):
    # ls = get_languages(ms, languages=languages, threshold=0)
    # values=[]
    # labels=[]

    # for l in ls:
        # f = ls[l] * 100
        # values.append(f)
        # labels.append(f"{l} ({round(f)}%)")

    # patches, texts = ax.pie(values, labels=labels, startangle=140)

    # #labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

    # # sort_legend = True
    # # if sort_legend:
        # # patches, labels, dummy =  zip(*sorted(zip(patches, labels, values),
                                              # # key=lambda x: x[2],
                                              # # reverse=True))

    # # ax.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.1, 1.),
               # # fontsize=8)

# def plot_languages(ms, ax, languages=None):
    # ls = get_languages(ms, languages=languages, threshold=0)
    # values=[]
    # labels=[]

    # for l in ls:
        # f = ls[l] * 100
        # values.append(f)
        # labels.append(f"{l} ({round(f)}%)")


def get_info(chat, languages=None):
    if not IPython: raise RuntimeError("IPython is missing")

    languages = ""
    for l, f in get_languages(chat, languages).items():
        languages += f"{l}: {100 * f:.0f}%, "
    languages = languages[:-2]


    table = [('Length', len(chat)),
             ('Languages', languages)]
             
    html = "<table>"
    for row in table:
        html += "<tr>"
        for col in row:
            html += f"<td>{col}</td>"
        html += "</tr>"
    html += "</table>"
    IPython.display.display(IPython.display.HTML(html))

def spans_overlap(span1, span2):
    a1, b1 = span1
    a2, b2 = span2
    if a1 < a2 and b1 > a2: return True
    if a1 < b2 and b1 > b2: return True
    if a2 < a1 and b2 > a1: return True
    if a2 < b1 and b2 > b1: return True
    return False

def replace_body_inplace(message, span, target):
    body = message.body[:span[0]] + \
            target + \
            message.body[span[1]:]
    message.body = body
    dlen = len(target) - (span[1] - span[0])
    for i in message.issues:
        if i.span[0] > span[1]: i.span[0] += dlen
        if i.span[1] > span[1]: i.span[1] += dlen

def censor(message: Message,
           pattern: str,
           replacer: Optional[Callable] = None, 
           tag: Optional[str] = None) -> Message:

    message = copy.deepcopy(message)
    if isinstance(pattern, list):
        pattern = '|'.join(re.escape(p) for p in pattern)

    matcher = re.compile(pattern, re.IGNORECASE)
    for match in matcher.finditer(message.body):
        span = match.span()

        overlap = False
        for i in message.issues:
            if spans_overlap(span, i.span):
                overlap = True

        if overlap: continue

        if replacer:
            target = replacer(match.group())
            replace_body_inplace(message, match.span, target)
            span = (span[0], span[0] + len(target))
        else:
            pass

        i = Issue(span=span, type='censor')
        if tag: i.data['tag'] = tag
        message.issues.append(i)

    return message

def make_name_mapper(users: list[str], fixed: dict = {}):
    import faker
    fkr = faker.Faker()
    mapper = {}
    for u in users:
        if u in fixed:
            mapper[u] = fixed[u]
        else:
            mapper[u] = fkr.name()

    # try lowercase
    temp = {}
    for s, t in mapper.items():
        temp[s.lower()] = t.lower()
    mapper |= temp

    # try also partial substitutions
    temp = {}
    for s, t in mapper.items():
        ss = [i for i in s.split() if i]
        tt = [i for i in t.split() if i]
        for i in range(len(ss)):
            temp[ss[i]] = tt[min(i, len(tt) - 1)]
    mapper |= temp

    mapper2 = {}
    for s, t in mapper.items():
        if len(s) <= 3: continue
        mapper2[s] = t
    return mapper2

def change_names(message: Message, mapper: dict) -> Message:
    """ Must be done BEFORE censorship """

    m = copy.deepcopy(message)
    for s, t in mapper.items():
        m.user = m.user.replace(s, t)
        m.body = m.body.replace(s, t)
        for r in m.reactions:
            r.user = r.user.replace(s, t)
    return m

def chat_to_actions(chat: list[Message]) -> list[Action]:
    """ Watch out for reaction time information """

    actions = []
    def append(*args, **kwargs): actions.append(Action(*args, **kwargs))

    for m in chat:
        append(time=m.time, 
               user=m.user, 
               type='message', 
               data={'body': m.body,
                     'issues': m.issues})
        for r in m.reactions:
            append(time=m.time, user=r.user, type='reaction', data={"name": r.name})

    return actions

def actions_to_chat(actions: list[Action]) -> list[Message]:
    """ Watch out for reaction time information """

    messages = []
    for a in actions:
        if a.type == 'message':
            messages.append(Message(time=a.time, 
                                    user=a.user, 
                                    body=a.data['body'],
                                    issues=a.data.get('issues', [])))
        elif a.type == 'reaction':
            if len(messages) == 0: continue
            messages[-1].reactions.append(Reaction(user=a.user, name=a.data['name']))
        else:
            pass
    return messages

def simulate(actions: list[Action], sleep: Optional[float] = None):
    if not IPython: raise RuntimeError("IPython is missing")

    def render(actions):
        chat = actions_to_chat(actions)
        IPython.display.clear_output(wait=True)
        make_chat_view(chat, range(len(chat)))

    if sleep:
        for i in range(1, len(actions)):
            render(actions[:i])
            time.sleep(sleep)
    else:
        render(actions)

def group_by_user_and_time(xs: list, duration_limit=None, pause_limit=None, count_limit=None) -> list[list]:
    buff = []
    last = None
    group = []
    for x in xs:
        if \
                last is None or \
                group[0].user != x.user or \
                (duration_limit is not None and x.time - group[0].time > duration_limit) or \
                (pause_limit is not None and x.time - group[-1].time > pause_limit) or \
                (count_limit is not None and len(group) >= count_limit):
                    if group: buff.append(group)
                    group = []

        group.append(x)
        last = x

    if group: buff.append(group)
    return buff

def mask_actions(actions: list[Action], predicate: Callable) -> list[Action]:
    def mapper(a):
        a1 = copy.deepcopy(a)
        a1.data['mask_action'] = predicate(a)
        return a1
    return list(map(mapper, actions))

def add_control_actions(actions: list[Action], 
                        users, 
                        duration_limit=None, 
                        pause_limit=None, 
                        count_limit=None,
                        idle_rate=1) -> list[Action]:
    buff = []

    for group in group_by_user_and_time(actions, 
                                        duration_limit=duration_limit, 
                                        pause_limit=pause_limit,
                                        count_limit=count_limit):
        user = group[0].user
        #buff.append(Action(time=0, user='', type='sep'))
        if user in users:
            for i, a in enumerate(group):
                a1 = copy.deepcopy(a)
                if a1.type == 'message':
                    if i == 0 or group[i - 1].type != 'message':
                        a1.data['types'] = True
                if i != 0:
                    a1.data['followup'] = True
                if i == len(group) - 1:
                    a1.data['eos'] = True
                buff.append(a1)

        else:
            for i, a in enumerate(group):
                a1 = copy.deepcopy(a)
                if i != 0:
                    if np.random.uniform() < idle_rate:
                        for u in users:
                            buff.append(Action(time=a1.time, user=u, type='idle'))
                buff.append(a1)

    return buff

def view_masked(s: str, m: str):
    if not IPython: raise RuntimeError("IPython is missing")

    buff = ""
    mask = False
    last = 0

    i = 0
    for i in range(len(m)):
        tag = None
        if m[i] and not mask:
            tag = """<span style="color: white">"""
        if not m[i] and mask:
            tag = """</span>"""

        if tag is None: continue
        buff += s[last:i] + tag
        last = i
        mask = m[i]

    if i != last:
        buff += s[last:i]

    html = f"""<pre style="color: grey">{buff}</pre>"""
    IPython.display.display(IPython.display.HTML(html))

def action_to_string(action: Action, 
                     return_mask: bool = False,
                     special_tokens: dict = {}):
    if action.type == 'message':
        b = action.data['body']
        m = ['m'] * len(b)
        for i in action.data['issues']:
            if i.type in ['censor']:
                for j in range(i.span[0], i.span[1]):
                    m[j] = ' '
        m = ''.join(m)

        if action.data.get('types', False):
            s = f"{special_tokens['types']}\n{b}"
            m = m.rjust(len(s), 'm')
        else:
            m = m
            s = b
    elif action.type == 'reaction':
        s = f"{special_tokens['reacts']} {action.data['name']}"
        m = 'm' * len(s)
    elif action.type == 'idle':
        m = ""
        s = f""
    elif action.type == 'sep':
        m = ""
        s = f"================================================"
    else: 
        raise NotImplementedError

    if action.type not in ['sep']:
        if not action.data.get('followup', False):
            s = f"{action.user}: {s}"
            m = m.rjust(len(s), 'm')

    s += f"{special_tokens['eos']}\n" if action.data.get('eos', False) else '\n'
    m = m.ljust(len(s), 'm')

    if action.data['mask_action']:
        pass
    else:
        m = " " * len(s)

    m = [True if c == 'm' else False for c in m]
    return (s, m) if return_mask else s

def action_from_string(s: str) -> Action:
    time = 0
    user = ""
    type = ""
    data = {}

    if m := re.search(r"^(.+)(?=:)", s):
        user = m.group(0)
        s = s[len(user)+1:].lstrip()
    else:
        data['followup'] = True

    if m := re.match(r'\<react\>', s):
        type = 'reaction'
    elif s.strip() == '':
        type = 'idle'
    else:
        type = 'message'
        if m := re.match(r'\<types\>', s):
            data['types'] = True

    if s.rstrip().endswith('<eos>'):
        data['eos'] = True
        if type == 'message': data['body'] = s.rstrip()[:-5].rstrip()
    else:
        if type == 'message': data['body'] = s.rstrip()

    return Action(time=time, user=user, type=type, data=data)

def tokenize_action(action: Action, tokenizer, special_tokens):
    string, mask = action_to_string(action, 
                                    return_mask=True, 
                                    special_tokens=special_tokens)
    out = tokenizer.encode(string)
    out_ids = np.array(out)

    def compute_offsets():
        offsets = []
        for i in range(len(out_ids)):
            offsets.append(len(tokenizer.decode(out_ids[:i])))
        for i in range(len(offsets)):
            offsets[i] = min(offsets[i:])

        return offsets

    offsets = compute_offsets()
    mask_out = np.array([False] * len(out_ids))

    source_prev = 0
    target_prev = 0
    for target_p, source_p in enumerate(offsets + [len(mask) + 2]):
        if source_prev == source_p: continue

        if np.all(mask[source_prev : source_p]):
            mask_out[target_prev : target_p] = True

        source_prev = source_p
        target_prev = target_p

    return {'ids': out_ids,
            'mask': mask_out}

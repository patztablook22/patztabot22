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

def get_users(chat: list) -> list[str]:
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
                    <td style="text-align: left;"><span style="white-space: pre">{content}</span>{reactions}</td></tr>"""

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
            classified += 1
            l, _ = langid.classify(body)
        else:
            l = None
        buff.append(l)
    
    return buff if batch else buff[0]

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
    from unidecode import unidecode
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

    # remove diacritics
    temp = {}
    for s, t in mapper.items():
        temp[unidecode(s)] = unidecode(t)
    mapper |= temp

    return mapper

def change_names(message: Message, mapper: dict) -> Message:
    """ Must be done BEFORE censorship """

    m = copy.deepcopy(message)
    for s, t in mapper.items():
        if len(s) <= 3:
            if m.user == s: m.user = t
            for r in m.reactions:
                if r.user == s: r.user = t
        else:
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
        if len(m.body) > 0 or len(m.attachments) == 0:
            append(time=m.time, 
                   user=m.user, 
                   type='message', 
                   data={'body': m.body,
                         'issues': m.issues})
        for a in m.attachments:
            append(time=m.time, user=m.user, type='attachment', data={'name': a})
        for r in m.reactions:
            append(time=m.time, user=r.user, type='reaction', data={'name': r.name})

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
        elif a.type == 'attachment':
            messages.append(Message(time=a.time, 
                                    user=a.user, 
                                    body="",
                                    attachments=[a.data['name']],
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
        a1.data['mask_action'] = a1.data.get('mask_action', False) or predicate(a)
        return a1
    return list(map(mapper, actions))

def add_control_actions(actions: list[Action], 
                        agents, 
                        duration_limit=None, 
                        pause_limit=None, 
                        count_limit=None,
                        idle_rate=1) -> list[Action]:
    buff = []
    prev_group = None

    for group in group_by_user_and_time(actions, 
                                        duration_limit=duration_limit, 
                                        pause_limit=pause_limit,
                                        count_limit=count_limit):
        user = group[0].user
        if user in agents:
            for i, a in enumerate(group):
                a1 = copy.deepcopy(a)
                if i != 0:
                    a1.data['followup'] = True
                if i == len(group) - 1: 
                    a1.data['eos'] = True
                buff.append(a1)

        else:
            for i, a in enumerate(group):
                a1 = copy.deepcopy(a)
                if i != 0 or (prev_group and prev_group[0].user == user):
                    if np.random.uniform() < idle_rate:
                        for agent in agents:
                            buff.append(Action(time=a1.time, 
                                               user=agent, 
                                               type='idle',
                                               data={'eos': True}))
                buff.append(a1)
        prev_group = group
    return buff

def view_masked(s: str, m: str):
    if not IPython: raise RuntimeError("IPython is missing")

    buff = ""
    mask = True
    last = 0

    i = 0
    for i in range(len(m)):
        tag = None
        if not m[i] and mask:
            tag = """<span style="color: white">"""
        if m[i] and not mask:
            tag = """</span>"""

        if tag is None: continue
        buff += s[last:i] + tag
        last = i
        mask = m[i]

    if i != last:
        buff += s[last:i+1]

    html = f"""<pre style="color: grey; white-space: pre; width: 100%">{buff}</pre>"""
    IPython.display.display(IPython.display.HTML(html))

CONTROL_TOKENS = {'goes': '<|goes|>',
                  'eos': '<|endoftext|>',
                  'eoa': '<|endofaction|>'}

for action_type in ['message', 'reaction', 'idle', 'attachment']:
    CONTROL_TOKENS[action_type] = f"<|{action_type}|>"

def action_to_string(action: Action, 
                     return_mask: bool = False,
                     control_tokens: dict = {}):
    control_tokens = control_tokens | CONTROL_TOKENS
    if action.type == 'message':
        b = action.data['body']
        m = [' '] * len(b)
        for i in action.data.get('issues', []):
            if i.type in ['censor']:
                for j in range(i.span[0], i.span[1]):
                    m[j] = 'm'
        m = ''.join(m)
        s = b

    elif action.type == 'reaction':
        s = f"{action.data['name']}"
        m = ' ' * len(s)
    elif action.type == 'idle':
        s = ""
        m = f' ' * len(s)
    elif action.type == 'attachment':
        s = f"{action.data['name']}"
        m = ' ' * len(s)
    else: 
        raise NotImplementedError

    s = f"{control_tokens[action.type]}{s}"
    m = m.rjust(len(s), ' ')

    if not action.data.get('followup', False):
        s = f"{action.user}{control_tokens['goes']}{s}"
        m = m.rjust(len(s), ' ')

    s += control_tokens['eoa']
    m = m.ljust(len(s), ' ')

    if action.data.get('eos', False):
        s += control_tokens['eos']
        m = m.ljust(len(s), ' ')

    if action.data.get('mask_action', False):
        m = "m" * len(s)
    else:
        pass

    m = [False if c == ' ' else True for c in m]
    return (s, m) if return_mask else s

def yeet_actions(s: str, control_tokens: dict = {}) -> list[Action]:
    #print(s)
    control_tokens = control_tokens | CONTROL_TOKENS
    time = 0
    user = None
    type = None
    data = {'yeet': {}}

    goes = CONTROL_TOKENS['goes'] 
    if (pos := s.find(goes)) != -1:
        user, s = s[:pos], s[pos + len(goes):]

    types = ['message', 'reaction', 'idle', 'eos']
    for t in types:
        if s.startswith(CONTROL_TOKENS[t]):
            type = t
            s = s[len(CONTROL_TOKENS[t]):]
            break

    if not user and not type: return []

    eoa = CONTROL_TOKENS['eoa']
    if (pos := s.find(eoa)) != -1:
        s, rest = s[:pos], s[pos + len(eoa):]
        data['yeet']['eoa'] = True
    else:
        data['yeet']['eoa'] = False
        rest = None

    for tok in control_tokens.values():
        s = s.replace(tok, '')
    s = s.strip()

    if not type and s: type = 'message'

    if type == 'eos':
        pass
    elif type == 'idle':
        pass
    elif type == 'message':
        data['body'] = s
    elif type == 'reaction':
        data['name'] = s
        
    buff = [Action(time=time, user=user, type=type, data=data)]
    if rest: buff += yeet_actions(rest, control_tokens=control_tokens)
    return buff

def tokenize_action(action: Action, tokenizer, control_tokens: dict = {}):
    control_tokens = control_tokens | CONTROL_TOKENS
    controL_tokens_ids = {k: tokenizer(v).input_ids[0] for k, v in control_tokens.items()}
    string, mask = action_to_string(action, 
                                    return_mask=True, 
                                    control_tokens=control_tokens)
    out = tokenizer.encode(string)
    out_ids = np.array(out)

    def compute_offsets():
        offsets = []
        extra = 0
        prev_extra = False
        #print("".ljust(5), f"|{string}|")
        for i in range(len(out_ids)):
            temp = tokenizer.decode(out_ids[:i])
            if i > 0 and out_ids[i-1] in controL_tokens_ids.values():
                extra += 1
                temp += ' '
                if not prev_extra: extra += 1
                prev_extra = True
            else:
                prev_extra = False
            offsets.append(len(temp) - extra)
            #print(str(offsets[-1]).ljust(5), f"|{tokenizer.decode(out_ids[:i])}|")
        for i in range(len(offsets)):
            offsets[i] = min(offsets[i:])

        return offsets

    offsets = compute_offsets()
    mask_out = np.array([False] * len(out_ids))

    source_prev = 0
    target_prev = 0
    for target_p, source_p in enumerate(offsets + [len(mask) + 2]):
        #print('source', source_prev, source_p, string[source_prev:source_p])
        #print('target', target_prev, target_p, out_ids[target_prev:target_p])
        if np.any(mask[source_prev : source_p]):
            mask_out[target_prev : target_p] = True

        if source_prev == source_p: continue

        source_prev = source_p
        target_prev = target_p

    return {'ids': out_ids,
            'mask': mask_out}

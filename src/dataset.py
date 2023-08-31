import os, json, time, re
from torch.utils.data import Dataset
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    author: str
    timestamp: float
    content: Optional[str]

class Conversation:
    def __init__(self, title=None, participants=None, messages=None):
        if participants is None: participants = set()
        if messages is None: messages = []
        self.title = title
        self.participants = participants
        self.messages = messages

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx]

    def __setitem__(self, idx, val):
        self.messages[idx] = val

    def __iter__(self):
        for m in self.messages:
            yield m

    def __repr__(self):
        return f'Conversation "{self.title}"'

    def head(self, n=10):
        Conversation.dump(self.messages[:n])

    def example(self, n=10):
        begin = np.random.randint(0, max(len(self.messages) - n, 1))
        Conversation.dump(self.messages[begin : begin + n])

    @classmethod
    def dump(cls, messages, line_width=96):
        name_pad = max([len(m.author) for m in messages])
        w = line_width - name_pad
        for m in messages:
            print(m.author.rjust(name_pad) + ': ', end='')
            first = True
            if not m.content:
                print()
                continue
            for line in m.content.splitlines():
                continues = False
                for begin in range(0, len(line), w):
                    l = line[begin : begin + w]
                    if first:
                        first = False
                    elif continues:
                        print(' ' * (name_pad - 2) + '... ', end='')
                    else:
                        print(' ' * (name_pad + 2), end='')
                    print(l)
                    continues = True

def load_simulated_conversations(root):
    cs = []
    for file in os.listdir(root):
        cs.append(load_simulated_conversation(os.path.join(root, file)))
    return cs

def mark_avoid(conversation, pattern, special_tokens):
    tok = special_tokens['avoid']
    def ma(message):
        if not message.content: return message
        newc = re.sub(rf'(?P<avoid>{pattern})',
                      rf'{tok}\g<avoid>',
                      message.content)
        if newc == message.content:
            return message
        else:
            return Message(message.author, message.timestamp, newc)

    return Conversation(conversation.title,
                        conversation.participants,
                        list(map(ma, conversation.messages)))

def load_simulated_conversation(path):
    c = Conversation(title=f"{os.path.basename(path).split('.')[0]} (simulated)")
    t = time.time()
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        author = line.split(':')[0].strip()
        content = ':'.join(line.split(':')[1:]).strip()
        m = Message(author, t, content)
        t += 5
        c.participants.add(author)
        c.messages.append(m)
    return c

def unfuck(s):
    return s.encode('latin1').decode('utf8')

def load_messenger_conversation(root):
    jsons = list(filter(lambda x: x.endswith('json'), os.listdir(root)))
    conv = Conversation()

    messages = []
    for j in jsons:
        with open(os.path.join(root, j), 'r') as f:
            data = json.load(f)

        for p in data['participants']:
            p['name'] = unfuck(p['name'])

        for m in data['messages']:
            author = unfuck(m['sender_name'])
            if 'content' in m:
                content = unfuck(m['content'])
            else:
                content = None
            timestamp = m['timestamp_ms'] / 1000
            messages.append(Message(author, timestamp, content))

        for p in data['participants']:
            conv.participants.add(p['name'])

        data['title'] = unfuck(data['title'])
        conv.title = data['title']

    conv.messages = sorted(messages, key=lambda m: m.timestamp)
    return conv

def rename_conversation_user(conversation, user_original, user_target):
    c = Conversation()
    c.title = conversation.title.replace(user_original, user_target)
    for p in conversation.participants:
        c.participants.add(user_target if p == user_original else p)
    for m in conversation.messages:
        author = user_target if m.author == user_original else m.author
        if m.content is None:
            content = None
        else:
            content = m.content.replace(user_original, user_target)
        c.messages.append(Message(author, m.timestamp, content))
    return c

def load_messenger_conversations(path):
    inbox = os.listdir(os.path.join(path, 'inbox'))
    archive = os.listdir(os.path.join(path, 'archived_threads'))
    conversations = [os.path.join(path, 'inbox', i) for i in inbox] \
                  + [os.path.join(path, 'archived_threads', a) for a in archive]
    return [load_messenger_conversation(c) for c in conversations]

def generate_corpus(file, conversations, special_tokens, break_tollerance_s):
    def dump_conversation(file, conversation):
        def continues(m0, m):
            if m0 is None or m0.author != m.author \
                    or not m0.content or not m.content \
                    or m.timestamp - m0.timestamp > break_tollerance_s:
                        return False
            else:
                return True

        file.write(special_tokens['conversation_start'] + '\n')
        m0 = None
        for m in conversation:
            if continues(m0, m):
                file.write(special_tokens['breakpoint'] + '\n' + m.content)
            else:
                if m0 is not None:
                    file.write(special_tokens['message_end'] + special_tokens['breakpoint'] + '\n')
                if m.content:
                    file.write(special_tokens['message_start'] + m.author + \
                            special_tokens['writes'] + m.content)
            m0 = m

        file.write(special_tokens['message_end'] + special_tokens['breakpoint'] + \
            '\n' + special_tokens['conversation_end'] + '\n')

    with open(file, 'w') as f:
        for c in conversations:
            dump_conversation(f, c)

def merge_adjacent_messages(conversation, time_tollerance_s):
    new = Conversation()
    new.title = conversation.title
    new.participants = conversation.participants
    
    m0 = None
    for m in conversation:
        if m0 is None or m0.author != m.author \
                or not m0.content or not m.content \
                or m.timestamp - m0.timestamp > time_tollerance_s:
            if m0 is not None:
                new.messages.append(m0)
            m0 = Message(m.author, m.timestamp, m.content)
        else:
           m0.content += "\n" + m.content
    return new

def filter_messages_by_content(conversation, predicate):
    new = Conversation()
    new.title = conversation.title
    new.participants = conversation.participants
    new.messages = list(filter(lambda m: predicate(m.content), conversation.messages))
    return new

class ChatDataset(Dataset):
    @classmethod
    def generate_mask(cls, text, tokenizer, whitelist, special_tokens):
        mstart = tokenizer(special_tokens['message_start']).input_ids[0]
        writes = tokenizer(special_tokens['writes']).input_ids[0]
        mend = tokenizer(special_tokens['message_end']).input_ids[0]
        mask = np.zeros(len(text))
        mstarts = np.where(text == mstart)[0]
        writess = np.where(text == writes)[0]
        mends = np.where(text == mend)[0]
        for ms, wr, me in zip(mstarts, writess, mends):
            nick = tokenizer.decode(text[ms+1 : wr])
            if nick in whitelist:
                mask[ms:me+1] = 1  
        return mask
        
    def __init__(self, text_path, tokenizer, block_size, whitelist, special_tokens):
        assert os.path.isfile(text_path), f"Input file path {text_path} not found"
        self.examples = []
        with open(text_path, encoding='utf-8') as f:
            text = f.read()
        tokenized_text = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
        mask = ChatDataset.generate_mask(tokenized_text, tokenizer, whitelist, special_tokens)
        begin = 0
        i = 0
        longest = 0
        while begin < len(tokenized_text):
            if isinstance(block_size, int):
                bs = block_size
            else:
                bs = block_size(i)
            self.examples.append((tokenized_text[begin : begin + bs], mask[begin : begin + bs]))
            longest = max(longest, bs)
            begin += bs

        pt = tokenizer(tokenizer.pad_token).input_ids[0]
        for i in range(len(self.examples)):
            t, m = self.examples[i]
            t = np.pad(t, (0, longest - len(t)), constant_values=pt)
            m = np.pad(m, (0, longest - len(m)), constant_values=0)
            self.examples[i] = (t,m)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        inputs, mask = self.examples[idx]
        labels = inputs.copy()
        labels[mask == 0] = -100
        return {
            'input_ids': torch.as_tensor(inputs).squeeze(),
            'attention_mask': torch.ones(inputs.shape).squeeze(),
            'labels': torch.as_tensor(labels).squeeze(),
        }

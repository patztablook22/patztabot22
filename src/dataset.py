import os, json
from torch.utils.data import Dataset
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

def unfuck(s):
    return s.encode('latin1').decode('utf8')

@dataclass
class Message:
    author: str
    timestamp_ms: int
    content: Optional[str]

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
            timestamp_ms = m['timestamp_ms']
            messages.append(Message(author, timestamp_ms, content))

        for p in data['participants']:
            conv.participants.add(p['name'])

        data['title'] = unfuck(data['title'])
        conv.title = data['title']

    conv.messages = sorted(messages, key=lambda m: m.timestamp_ms)
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
        c.messages.append(Message(author, m.timestamp_ms, content))
    return c

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

def load_messenger_conversations(path):
    inbox = os.listdir(os.path.join(path, 'inbox'))
    archive = os.listdir(os.path.join(path, 'archived_threads'))
    conversations = [os.path.join(path, 'inbox', i) for i in inbox] \
                  + [os.path.join(path, 'archived_threads', a) for a in archive]
    return [load_messenger_conversation(c) for c in conversations]

def generate_corpus(file, conversations, special_tokens):
    with open(file, 'w') as f:
        for conv in conversations:
            f.write(special_tokens['conversation_start'] + '\n')
            for msg in conv:
                if not msg.content: continue
                buff = special_tokens['message_start'] + msg.author \
                     + special_tokens['writes'] + msg.content \
                     + special_tokens['message_end'] + '\n'
                f.write(buff)
            f.write(special_tokens['conversation_end'] + '\n')

def merge_adjacent_messages(conversation, time_tollerance_s):
    new = Conversation()
    new.title = conversation.title
    new.participants = conversation.participants
    
    m0 = None
    for m in conversation:
        if m0 is None or m0.author != m.author \
                or not m0.content or not m.content \
                or (m.timestamp_ms - m0.timestamp_ms) / 1000 > time_tollerance_s:
            if m0 is not None:
                new.messages.append(m0)
            m0 = Message(m.author, m.timestamp_ms, m.content)
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
                mask[ms:me] = 1  
        return mask
        
    def __init__(self, text_path, tokenizer, block_size, whitelist, special_tokens):
        assert os.path.isfile(text_path), f"Input file path {text_path} not found"
        self.examples = []
        with open(text_path, encoding='utf-8') as f:
            text = f.read()
        tokenized_text = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
        mask = MessengerDataset.generate_mask(tokenized_text, tokenizer, whitelist, special_tokens)
        begin = 0
        i = 0
        while begin < len(tokenized_text):
            if isinstance(block_size, int):
                bs = block_size
            else:
                bs = block_size(i)
            self.examples.append((tokenized_text[begin : begin + bs], mask[begin : begin + bs]))
            begin += bs

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

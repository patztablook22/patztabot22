import os, json
from torch.utils.data import Dataset
import torch
import numpy as np

def unfuck(s):
    return s.encode('latin1').decode('utf8')

class Conversation:
    def __init__(self, root):
        self._root = root
        jsons = list(filter(lambda x: x.endswith('json'), os.listdir(root)))

        participants = set()
        messages = []
        title = None

        for j in jsons:
            with open(os.path.join(root, j), 'r') as f:
                data = json.load(f)

            for p in data['participants']:
                p['name'] = unfuck(p['name'])

            for m in data['messages']:
                m['sender_name'] = unfuck(m['sender_name'])
                if 'content' in m:
                    m['content'] = unfuck(m['content'])
                for r in m.get('reactions', []):
                    r['reaction'] = unfuck(r['reaction'])
                    r['actor'] = unfuck(r['actor'])

            for p in data['participants']:
                participants.add(p['name'])

            data['title'] = unfuck(data['title'])
            title = data['title']
            messages.extend(data['messages'])

        messages = sorted(messages, key=lambda x: x['timestamp_ms'])
        self.title = title
        self.participants = participants
        self.messages = messages

    def rename(self, user_original, user_target):
        self.title = self.title.replace(user_original, user_target)
        for p in self.participants:
            if p == user_original:
                self.participants.add(user_target)
                self.participants.remove(user_original)
        for m in self.messages:
            if m['sender_name'] == user_original:
                m['sender_name'] = user_target
            for r in m.get('reactions', []):
                if r['actor'] == user_original:
                    r['actor'] = user_target
            if 'content' in m:
                m['content'] = m['content'].replace(user_original, user_target)

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

def load_conversations(path):
    inbox = os.listdir(os.path.join(path, 'inbox'))
    archive = os.listdir(os.path.join(path, 'archived_threads'))
    conversations = [os.path.join(path, 'inbox', i) for i in inbox] \
                  + [os.path.join(path, 'archived_threads', a) for a in archive]
    return [Conversation(c) for c in conversations]

def generate_corpus(file, conversations, special_tokens):
    with open(file, 'w') as f:
        for conv in conversations:
            f.write(special_tokens['conversation_start'] + '\n')
            for msg in conv:
                if 'content' not in msg: continue
                buff = special_tokens['message_start'] + msg['sender_name'] \
                     + special_tokens['writes'] + msg['content'] \
                     + special_tokens['message_end'] + '\n'
                f.write(buff)
            f.write(special_tokens['conversation_end'] + '\n')

class MessengerDataset(Dataset):
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
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append((tokenized_text[i:i+block_size], mask[i:i+block_size]))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        inputs, mask = self.examples[idx]
        labels = inputs.copy()
        labels[mask == 0] = -100
        return torch.as_tensor(inputs), torch.as_tensor(labels)

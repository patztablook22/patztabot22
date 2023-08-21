import os
import json
from torch.utils.data import Dataset

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



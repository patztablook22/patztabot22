import torch
import dataclasses
from utils import datasets
import time
import shellbot
import importlib

def actions(chat, model, tokenizer, stop_token_ids, special_tokens):
    importlib.reload(datasets)

    actions = datasets.chat_to_actions(chat)
    buff = [datasets.action_to_string(a, special_tokens=special_tokens) for a in actions]
    prompt = '\n'.join(buff)
    input_ids = tokenizer(prompt).input_ids
    for i in range(5):
        response_ids = generate(input_ids, tokenizer, model, 
                                stop_token_ids=stop_token_ids)
        print(f"{response_ids=}", flush=True)

        response = tokenizer.decode(response_ids)
        print(f"{response=}", flush=True)

        action = datasets.action_from_string(response, special_tokens)
        print(f"{action=}", flush=True)

        if action.type == 'idle': break
        yield action
        if action.data.get('eos', False): break

    shellbot.success("generation terminated")

def generate(input_ids, tokenizer, model, **kwargs):
    input_ids = input_ids[-500:]
    i = kwargs.get('i', 0)
    length = len(input_ids)
    shellbot.log('generating', ...)
    output_ids = model.generate(input_ids=torch.tensor([input_ids], dtype=torch.long),
                                num_return_sequences=1,
                                eos_token_id=kwargs.get('stop_token_ids', None),
                                pad_token_id=tokenizer.pad_token_id,
                                min_length = length + 2,
                                max_length = length + 30,
                                do_sample=True,
                                temperature=3.0,
                                )[0]
    shellbot.success()

    response_ids = output_ids.tolist()[len(input_ids):]
    if i == 3: response_ids.append(tokenizer.eos_token_id)
    return response_ids

class FakeModel:
    def generate(self, input_ids, *args, **kwargs):
        time.sleep(1)
        return torch.tensor([input_ids.tolist()[0] + [4, 5, 6]], dtype=torch.long)

@dataclasses.dataclass
class FakeTokenizer:
    eos_token_id: int = 12345
    pad_token_id: int = 12345

    def __call__(self, *args, **kwargs):
        @dataclasses.dataclass
        class FakeResponse:
            input_ids: list
        return FakeResponse(input_ids=[1,2,3])


    def decode(self, *args, **kwargs):
        import random
        return random.choice([
            "hello there",
            "<|types|>",
            "<|reacts|> ❤️"
            ])

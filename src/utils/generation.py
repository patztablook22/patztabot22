import torch
import dataclasses
from utils import datasets
import time
import shellbot
import importlib
import threading
import numpy as np
from typing import Optional
import random
import queue

class ActionStreamer:
    def __init__(self, tokenizer, control_tokens: dict = {}):
        self._tokenizer = tokenizer
        self._token_cache = []
        self._next_tokens_are_prompt = True
        self._control_tokens = control_tokens | datasets.CONTROL_TOKENS
        self._eoa = tokenizer(self._control_tokens['eoa']).input_ids[0]
        self._queue = queue.Queue()
        self._end_signal = None

    def put(self, value):
        # if len(value.shape) > 1 and value.shape[0] > 1:
            # raise ValueError("Only batch size 1 is supporoted")
        # elif len(value.shape) > 1:
            # value = value[0]
        if len(value) > 1:
            raise ValueError("Only batch size 1 is supporoted")
        else:
            value = value[0]

        if self._next_tokens_are_prompt:
            self._next_tokens_are_prompt = False
            return

        if not isinstance(value, list):
            value = value.tolist()


        self._token_cache += value
        if self._eoa in self._token_cache:
            idx = self._token_cache.index(self._eoa)
            prev = self._token_cache[:idx + 1]
            new = self._token_cache[idx + 1:]
            to_yield = [prev, new]
            self._token_cache = new
            shift = True
        else:
            to_yield = [self._token_cache]
            shift = False

        for ids in to_yield:
            text = self._tokenizer.decode(ids)
            actions = datasets.yeet_actions(text, control_tokens=self._control_tokens)
            if not shift: actions = actions[-1:]
            for a in actions: self._queue.put(a)

    def end(self):
        self._queue.put(self._end_signal)

    def __iter__(self):
        return self

    def __next__(self):
        v = self._queue.get()
        if v == self._end_signal:
            raise StopIteration
        else:
            return v

class Pipeline:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 agents: list = [], 
                 control_tokens: dict = {}):
        self._model = model
        self._tokenizer = tokenizer
        self._control_tokens = control_tokens

    def _make_prompt(self, chat: list[datasets.Message]) -> str:
        actions = datasets.chat_to_actions(chat)
        buff: list = [datasets.action_to_string(a) for a in actions]
        prompt = '\n'.join(buff)
        return prompt

    def __call__(self, chat: list[datasets.Message]):
        prompt = self._make_prompt(chat)
        inputs = self._tokenizer(prompt, return_tensors='pt')
        length = inputs.input_ids.shape[-1]
        streamer = ActionStreamer(tokenizer=self._tokenizer)

        gen_kwargs = dict(inputs, streamer=streamer)
        gen_kwargs |= dict(
            pad_token_id=self._tokenizer.pad_token_id,
            min_length = length + 2,
            max_length = length + 30,
            do_sample=True,
            temperature=3.0,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for action in streamer:
            yield action




class SamplingModel:
    def __init__(self, 
                 tokenized: Optional[list[list[int]]] = None, 
                 untokenized: Optional[list[str]] = None,
                 tokenizer = None,
                 sleep: float = 0.1):

        if tokenized:
            self._data = tokenized
        else:
            if tokenizer is None or untokenized is None:
                raise ValueError("No data to sample")
            self._data = tokenizer(untokenized).input_ids

        self._sleep = sleep

    def generate(self, input_ids, *args, streamer=None, **kwargs):
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()

        if len(input_ids) != 1:
            raise ValueError("Only batch size 1 is supported")

        generated = input_ids[0]
        sample = random.choice(self._data)

        if streamer is not None:
            streamer.put([generated])

        for i in sample:
            time.sleep(self._sleep)
            generated.append(i)
            if streamer is not None:
                streamer.put([[i]])

        if streamer is not None:
            streamer.end()

        return [generated]








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

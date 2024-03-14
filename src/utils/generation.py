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
    def __init__(self,
                 tokenizer, 
                 control_tokens: dict = {},
                 debug: bool = False):
        self._tokenizer = tokenizer
        self._token_cache = []
        self._next_tokens_are_prompt = True
        self._control_tokens = datasets.CONTROL_TOKENS | control_tokens
        self._eoa = tokenizer(self._control_tokens['eoa']).input_ids[0]
        self._queue = queue.Queue()
        self._end_signal = None
        self._clue = ""
        self._debug = debug

    def put(self, value):
        # if len(value.shape) > 1 and value.shape[0] > 1:
            # raise ValueError("Only batch size 1 is supporoted")
        # elif len(value.shape) > 1:
            # value = value[0]
        if hasattr(value, 'shape'):
            if len(value) > 1:
                raise ValueError("Only batch size 1 is supporoted")
            value = value.tolist()[0]
        else:
            if len(value) > 1:
                raise ValueError("Only batch size 1 is supporoted")
            else:
                value = value[0]

        if isinstance(value, int):
            value = [value]

        if self._next_tokens_are_prompt:
            self._next_tokens_are_prompt = False
            prompt = self._tokenizer.decode(value)
            last_eoa = prompt.rfind(self._control_tokens['eoa'])
            if last_eoa == -1: return

            clue = prompt[last_eoa + len(self._control_tokens['eoa']):].strip()
            if clue.startswith(self._control_tokens['eos']):
                clue = clue[len(self._control_tokens['eos']):].strip()

            self._clue = clue
            return

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

        for i, ids in enumerate(to_yield):
            text = self._tokenizer.decode(ids)
            if i == 0: text = self._clue + text
            if self._debug:
                shellbot.log(text, ..., overwrite=True)
                if i != len(to_yield) - 1:
                    shellbot.success()
                    shellbot.log("")
            actions = datasets.yeet_actions(text, control_tokens=self._control_tokens)
            if not shift: actions = actions[-1:]
            for a in actions: 
                if a.type != 'idle': 
                    self._queue.put(a)

    def end(self):
        self._queue.put(self._end_signal)
        if self._debug:
            shellbot.success()
            shellbot.log("")

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
                 control_tokens: dict = {},
                 agents: list = [],
                 debug: bool = False,
                 action_type_whitelist: Optional[list[str]] = None,
                 action_type_blacklist: Optional[list[str]] = None,
                 continuous: bool = True):
        self._model = model
        self._tokenizer = tokenizer
        self._control_tokens = control_tokens | datasets.CONTROL_TOKENS
        self._control_tokens_ids = {k: tokenizer(v).input_ids[0] for k, v in self._control_tokens.items()}
        self._debug = debug
        self._agents = agents
        self._continuous = continuous
        self._action_type_whitelist = action_type_whitelist
        self._action_type_blacklist = action_type_blacklist
        self._action_type_whitelist = ['reaction']

    def chat_to_actions(self, chat: list[datasets.Message]) -> list[datasets.Action]:
        actions = datasets.chat_to_actions(chat)
        actions = datasets.add_control_actions(actions, 
                                               agents=self._agents, 
                                               duration_limit=180, 
                                               pause_limit=60)
        return actions

    def actions_to_prompt(self, actions: list[datasets.Action], agent_clue=None) -> str:
        buff: list = [datasets.action_to_string(a) for a in actions]
        prompt = ''.join(buff)

        goes = self._control_tokens['goes']
        if agent_clue == True or agent_clue is None and len(self._agents) == 1:
            if len(self._agents) != 1:
                raise ValueError("len(agents) must be 1 when clue=True is used, use str instead")
            prompt = prompt + list(self._agents)[0] + goes
        elif isinstance(agent_clue, str):
            prompt = prompt + agent_clue + goes

        return prompt

    def __call__(self, 
                 context: list[datasets.Message] | list[datasets.Action], 
                 agent_clue=None, 
                 action_type_whitelist: Optional[list[str]] = None,
                 action_type_blacklist: Optional[list[str]] = None,
                 continuous: Optional[bool] = None):
        if not context:
            actions: list[datasets.Action] = []
        elif "Message" in str(type(context[0])):
            actions: list[datasets.Action] = self.chat_to_actions(context)
        elif "Action" in str(type(context[0])):
            actions: list[datasets.Action] = context
        else:
            raise ValueError("`context` must be a list of messages or actions")

        do_whitelist = self._action_type_whitelist is not None or action_type_whitelist is not None
        whitelist = set(self._action_type_whitelist) if self._action_type_whitelist
        if action_type_whitelist: whitelist = whitelist.union(action_type_whitelist)

        do_blacklist = self._action_type_blacklist is not None or action_type_blacklist is not None
        blacklist = set(self._action_type_blacklist) if self._action_type_blacklist
        if action_type_blacklist: blacklist = blacklist.union(action_type_blacklist)

        all_types = set(['message', 'idle', 'reaction', 'attachment'])
        bad_types = all_types.difference(whitelist) if do_whitelist else set()
        if do_blacklist: bad_types = bad_types.union(blacklist)

        bad_words_ids = [self._control_tokens_ids[t] for t in bad_types]
        print(bad_words_ids, flush=True)

        prompt = self.actions_to_prompt(actions, agent_clue=agent_clue)
        if continuous is None: continuous = self._continuous
        if self._debug: print(prompt, flush=True)

        inputs = self._tokenizer(prompt, return_tensors='pt')
        inputs.input_ids = torch.tensor(inputs.input_ids.tolist()[0][:500])
        length = inputs.input_ids.shape[-1]
        streamer = ActionStreamer(tokenizer=self._tokenizer, debug=self._debug)

        gen_kwargs = dict(inputs, streamer=streamer)
        gen_kwargs |= dict(
            pad_token_id=self._tokenizer.pad_token_id,
            bad_words_ids=bad_words_ids,
            #min_length = length + 2,
            max_length = length + 128,
            do_sample=True,
            forced_eos_token_id = self._control_tokens_ids['eoa'],
            temperature=float(2.0),   # MUST be float
            top_k=16,
            top_p=0.7,
            repetition_penalty=0.95,
        )

        if self._debug: shellbot.log('generating...')
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for action in streamer:
            if not continuous:
                if not action.data['yeet']['eoa']: continue
            yield action
        if self._debug: shellbot.log('generation terminated')




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

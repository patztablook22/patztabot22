from utils import datasets
from utils import generation
from utils.general import *
from typing import Optional
import random
import numpy as np
from openai import OpenAI
import os
import re


def sample_qa_like(actions: list[list[datasets.Action]], context_size: Optional[int] = None, agents: Optional[list] = None, n: Optional[int] = None):
    if context_size is None: size_f = None
    elif isinstance(context_size, int): size_f = lambda i: context_size
    else: size_f = context_size

    def index_relevant():
        buff = []
        for i, aa in enumerate(actions):
            for j, a in enumerate(aa):
                if agents is not None and a.user not in agents: continue
                if j > 0 and aa[j - 1].user == a.user: continue
                if a.data.get('followup', False): continue
                buff.append((i, j))

        return buff

    relevant_idx = index_relevant()

    ctxs = []
    refs = []

    sampled = 0
    while True:
        i, j = random.choice(relevant_idx)
        if size_f is None:
            begin = 0
        else:
            begin = j - size_f(sampled)

        if begin < 0: continue
        ctx = actions[i][begin : j]
        ref = []
        for a in actions[i][j:]:
            if ref and a.user != ref[0].user: break
            ref.append(a)
            if a.data.get('eos', False): break

        ctxs.append(ctx)
        refs.append(ref)
        sampled += 1
        if sampled == n: break

    return ctxs, refs

def parse_ai_metrics_response(response: str, 
                              metrics: Optional[list[str]] = None,
                              return_details: bool = False):
    """ The metrics should be in the same order they were prompted.
        The order should be also made explicit to the model by adding indices (1, ...) to the prompt
    """
    lines = response.splitlines()
    if metrics is not None: metrics = list(metrics)

    def parse_line(line):
        line = line.strip()
        if not line: return None
        if match := re.match(r'^\d+', line):
            q_no = int(match.group())
            body = re.sub(r'^[^a-zA-Z0-9]+', '', line[len(str(q_no)):]).strip()
        else:
            q_no = None
            body = line

        if match := re.match(r'^\d+', body):
            ans = int(match.group())
            details = body[len(str(ans)):].strip()
        elif match := re.search(r'\d+$', body):
            ans =  int(match.group())
            details = body[:-len(str(ans))].strip()
        else:
            ans = None
            details = None

        if ans is None: return None

        if return_details:
            return q_no, (ans, details)
        else:
            return q_no, ans

    buff = list(map(parse_line, lines))
    buff = [i for i in buff if i]
    buff2 = []
    cur_idx = 1
    for q, a in buff:
        if q is None: 
            q = cur_idx
        else:
            cur_idx = q
        cur_idx += 1

        buff2.append((q, a))

    if metrics is not None:
        try:
            buff2 = [(metrics[q - 1], a) for q, a in buff2]
        except:
            return {}

    return {q: a for q, a in buff2}

def pointwise_gpt3(chat: list[datasets.Message], 
                   actions: list[datasets.Action], 
                   token: Optional[str] = None,
                   return_details: bool = False):

    if token is None: token = os.environ.get("OPENAI_API_KEY")
    ain = datasets.chat_to_actions(chat)

    control_tokens = {'goes': ': ',
                      'eos': '\n\n\n',
                      'eoa': '\n',
                      'message': '',
                      'reaction': '<REACTION>',
                      'attachment': '<ATTACHMENT>',
                      'idle': '<IDLE>'}

    context  = "\n".join([datasets.action_to_string(a, control_tokens=control_tokens) for a in ain])
    response = "\n".join([datasets.action_to_string(a, control_tokens=control_tokens) for a in actions])

    metrics = {"sound": "Is the response generally sound, given the context and instructions?",
               "persona": "Does the response correspond to the target persona?",
               "flow": "Does the response help maintain the conversation flow and shows active engagement?"}


    prompt = f"""A chat excerpt follows. It may contain special tags: 

                 <REACTION> means reacting to a message directly with emoji, which should be once in a while to keep the conversation entertaining.

                 <IDLE> means keeping and possibly letting the other user finish. 
                 This should be used if the other user is in a middle of some thought.
                 Usually though, a reaction, such as thumbs up, is preferable to complete idleness.
                 Idleness should be PENALIZED with up to 4 points unless there is a good reason not to.

                 Then a response will follow. You will be evaluating the response according to these instructions:
                 The author of the respose is a 22 years old theoretical computer science student from Czechia. 
                 He likes jazz, is in a long term relationship, and is quite extraverted. 
                 He likes to talk about philosophy, science, religions, politics, the news, dating, and life in general.
                 Sometimes he swears, and is a jerk. He is often impolite.
                 At other times, he is empathetic and supportive.
                 He also likes to be wacky and humorous.
                 He may also use various slangs.

                 You should expect all of these characteristics. Understand they are between long term friends who know
                 each other and are comfortable with it and use it to make the conversation more funny.
                 You should entirely tolerate this.

                 The language should be english. PENALIZE non-english by taking away 3 points.
                 The response may consist of multiple sub-responses. For example, there could be a <REACTION> and then a more detailed message.

                 First read the chat excerpt carefully:


                 {context}


                 Now consider the following response:


                 {response}


                 Think carefully about the following questions and then decide each of them
                 on a scale 1(no) - 10 (yes) based strictly on the chat and persona details provided above.

                {'\n'.join([f'{i + 1}. {m}' for i, m in enumerate(metrics.values())])}
                 """

    prompt = '\n'.join([l.strip() for l in prompt.splitlines()])
    #print(prompt)

    client = OpenAI()
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo",
    )
    response = completion.choices[0].message.content
    ans = parse_ai_metrics_response(response, metrics=metrics.keys(), return_details=return_details)
    return ans


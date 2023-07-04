import time
import genbot
import importlib

class GPT2(genbot.BatchedModel):
    """Vanilla pretrained OpenAI GPT-1, no custom finetuning or prompt engineering"""

    def __init__(self):
        from models.llm_cache import llm_cache
        if 'gpt2' not in llm_cache:
            #from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
            # tok = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            # mod = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
            from transformers import TFAutoModelWithLMHead, AutoTokenizer
            tok = AutoTokenizer.from_pretrained('gpt2')
            mod = TFAutoModelWithLMHead.from_pretrained('gpt2')
            # self.pipeline = pipeline('text-generation', model='openai-gpt',
            llm_cache['gpt2'] = (tok, mod)
                                 # max_new_tokens=20)

        tok, mod = llm_cache['gpt2']
        tok.padding_side = 'left'
        tok.pad_token = tok.eos_token

        self.tokenizer = tok
        self.model = mod
        super().__init__(max_size=1)

    def batch(self, feeds):
        enc = self.tokenizer(feeds, padding=True, return_tensors='tf')
        ids = self.model.generate(**enc,
                                  max_new_tokens=100,
                                  do_sample=True)
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)

        return texts

    async def __call__(self, channel, user):
        import models.context_prompt_pipeline as cppl
        importlib.reload(cppl)

        prompt = await cppl.context_prompt_pipeline(
            channel=channel,
            self_user=user,
            self_nick='Patrik',
            history_messages_limit=100,
            history_length_limit=1500,
            history_break='*break*',
            prefix='Patrik is a 21 years old student from Czechia. They have just met. When they started talking, Patrik was just vibing at home alone. They chatted for about an hour. The conversation went... \n',
            prefix_messages=[
                ('Patrik', 'i am just vibing at home'),
                ('Patrik', 'im kind of bored not gonna lie'),
                ('Patrik', 'anyway...')
            ],
            message_format=lambda nick,content: f'{nick} says: "{content}", \n',
            postfix='Patrik says: "'
        )

        if not prompt:
            return

        async with channel.typing():
            print(prompt)
            output: str = await self.call_batch(prompt)
            temp = output[len(prompt) - 5 :]
            pos = temp.find('says') + 4
            temp = temp[pos:]
            begin = temp.find('"') + 1

            new = temp[begin:]
            one = new[:new.find('"')].strip()

            return one

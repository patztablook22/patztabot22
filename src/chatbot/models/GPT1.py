import time
import genbot
import importlib

class GPT1(genbot.PromptPipeline):
    """Vanilla pretrained OpenAI GPT-1, no custom finetuning or prompt engineering"""

    BACKEND = 'pt'

    def __init__(self):
        from models.llm_cache import llm_cache
        if 'gpt1' not in llm_cache:
            #from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
            # tok = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            # mod = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
            if GPT1.BACKEND == 'pt':
                from transformers import AutoModelWithLMHead, AutoTokenizer
                tok = AutoTokenizer.from_pretrained('openai-gpt')
                mod = AutoModelWithLMHead.from_pretrained('openai-gpt')
            else:
                from transformers import TFAutoModelWithLMHead, AutoTokenizer
                tok = AutoTokenizer.from_pretrained('openai-gpt')
                mod = TFAutoModelWithLMHead.from_pretrained('openai-gpt')
            # self.pipeline = pipeline('text-generation', model='openai-gpt',
            llm_cache['gpt1'] = (tok, mod)
                                 # max_new_tokens=20)

        tok, mod = llm_cache['gpt1']
        tok.padding_side = 'left'
        tok.pad_token = 0
        #tok.add_special_tokens({'pad_token': '[PAD]'})

        self.tokenizer = tok
        self.model = mod
        super().__init__(batch_size_max=0)

    def process(self, prompts):
        print()
        print()
        print('#'*64)
        print()
        print()
        print('prompts', prompts)
        enc = self.tokenizer(prompts, padding=True, return_tensors=GPT1.BACKEND)
        print('enc', enc)
        ids = self.model.generate(enc, max_new_tokens=100)
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)

        return texts

    async def create_prompt(self, channel, user):
        import models.context_prompt_pipeline as cppl
        importlib.reload(cppl)

        prompt = await cppl.context_prompt_pipeline(
            channel=channel,
            self_user=user,
            self_nick='Patrik',
            history_messages_limit=6,
            history_length_limit=300,
            history_break='*break*',
            prefix='The conversation went... \n'
        )
        return prompt

    async def postprocess(self, channel, user, prompt, output):
        temp = output[len(prompt) - 5 :]
        pos = temp.find('said') + 4
        temp = temp[pos:]
        begin = temp.find('"') + 1

        new = temp[begin:]
        one = new[:new.find('"')].strip()
        print('asdfasdfasf')
        return one

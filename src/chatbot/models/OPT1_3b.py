import time
import genbot
import importlib

class OPT1_3b(genbot.BatchedModel):
    """Vanilla pretrained facebook OPT 1.3b version, no custom finetuning or prompt engineering"""

    def __init__(self):
        from models.llm_cache import llm_cache
        if 'gpt1' not in llm_cache:
            from transformers import TFAutoModelWithLMHead, AutoTokenizer
            tok = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
            mod = TFAutoModelWithLMHead.from_pretrained('facebook/opt-1.3b')
            llm_cache['gpt1.3b'] = (tok, mod)
                                 # max_new_tokens=20)
            tok.padding_side = 'left'
            tok.add_special_tokens({'pad_token': '[PAD]'})


        tok, mod = llm_cache['opt1.3b']

        self.tokenizer = tok
        self.model = mod
        super().__init__(max_size=1)

    def batch(self, feeds):
        enc = self.tokenizer(feeds, padding=True, return_tensors='tf')
        ids = self.model.generate(enc, max_new_tokens=100)
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)

        return texts

    async def __call__(self, channel, user):
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

        if not prompt:
            return

        async with channel.typing():
            print(prompt)
            output: str = await self.call_batch(prompt)
            temp = output[len(prompt) - 5 :]
            pos = temp.find('said') + 4
            temp = temp[pos:]
            begin = temp.find('"') + 1

            new = temp[begin:]
            one = new[:new.find('"')].strip()

            return one


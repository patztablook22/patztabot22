import time
import genbot
import importlib

class GPT1(genbot.PromptPipeline):
    """Vanilla pretrained OpenAI GPT-1, no custom finetuning or prompt engineering"""

    GEN_PARAMS = {
        'temperature': 1.1,
        'top_p': 0.9,
        'repetition_penalty': 1.35,
        'top_k': 50,
        'do_sample': True,
        'typical_p': 0.2,
        'max_new_tokens': 50,
    }

    def __init__(self):
        from models.llm_cache import llm_cache
        if 'openai-gpt' not in llm_cache:
            from transformers import pipeline
            pl = pipeline('text-generation',
                          model='openai-gpt',
                          **GPT1.GEN_PARAMS)

            llm_cache['openai-gpt'] = pl

        self.pipeline = llm_cache['openai-gpt']

        super().__init__(batch_size_max=0)

    def process(self, prompts):
        outputs = self.pipeline(prompts, **GPT1.GEN_PARAMS)
        #print('=' * 30)
        #print(outputs)
        gens = [i[0]['generated_text'] for i in outputs]
        print('=' * 30)
        print(gens)
        return gens

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
            prefix='The chat went... \n',
            message_format=lambda nick,content: f'{nick}: "{content}", \n',
            postfix='Patrik: "'
        )
        return prompt

    async def postprocess(self, channel, user, prompt, output):
        temp = output[len(prompt) - 8 :]
        pos = temp.find('Patrik') + 6
        temp = temp[pos:]
        begin = temp.find('"') + 1

        new = temp[begin:]
        one = new[:new.find('"')].strip()
        return one

#from transformers import pipeline

class Gpt2:
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
        model='gpt2-xl'
        model='openai-gpt'

        # pl = pipeline('text-generation',
                      # model=model,
                      # **Gpt2.GEN_PARAMS)
        # self._pipeline = pl


    def predict(self, prompts):
        # outputs = self._pipeline(prompts, **Gpt2.GEN_PARAMS)
        # gens = [i[0]['generated_text'] for i in outputs]
        outputs = ["pong: " + p for p in prompts]
        return outputs


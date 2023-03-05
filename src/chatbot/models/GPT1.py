from Model import Model

class GPT1(Model):
    def __init__(self):
        from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
        from transformers import pipeline, set_seed
        self.pipeline = pipeline('text-generation', model='openai-gpt')

    def __call__(self, context):
        return self.pipeline(context[-512:])[0]['generated_text']

    def kill(self):
        self.pipeline = None

def load():
    return GPT1()

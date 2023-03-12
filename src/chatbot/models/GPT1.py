import genbot

@genbot.model
class GPT1:
    """Vanilla pretrained OpenAI GPT-1, no custom finetuning or prompt engineering"""

    def __init__(self):
        from transformers import pipeline
        self.pipeline = pipeline('text-generation', model='openai-gpt')
        pass

    async def __call__(self, message):
        async with message.channel.typing():
            context = message.content
            return self.pipeline(context[-512:])[0]['generated_text']

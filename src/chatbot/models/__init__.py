import genbot
import importlib

@genbot.model(name='GPT1')
def gpt1():
    """Vanilla pretrained OpenAI GPT-1, no custom finetuning or prompt engineering"""

    import models.GPT1 as GPT1
    importlib.reload(GPT1)
    return GPT1.GPT1()




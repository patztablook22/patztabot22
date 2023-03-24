import genbot
import importlib

@genbot.model(name='GPT-1', default=True)
def gpt1():
    """Vanilla pretrained OpenAI GPT-1, no custom finetuning or prompt engineering"""

    async def reload_run(channel, user):
        try: 
            import models.GPT1 as GPT1
            importlib.reload(GPT1)
            target = GPT1.GPT1()
            return await target(channel, user)
        except Exception as e:
            return str(e)

    return reload_run


@genbot.model(name='OPT-1.3b')
def opt1_3b():
    """Vanilla pretrained facebook OPT 1.3b version, no custom finetuning or prompt engineering"""

    async def reload_run(channel, user):
        try: 
            import models.OPT1_3b as OPT1_3b
            importlib.reload(OPT1_3b)
            target = OPT1_3b.OPT1_3b()
            return await target(channel, user)
        except Exception as e:
            return str(e)

    return reload_run

@genbot.model(name='GPT-2')
def gpt2():
    """Vanilla pretrained facebook GPT-2, no custom finetuning or prompt engineering"""

    async def reload_run(channel, user):
        try: 
            import models.GPT2 as GPT2
            importlib.reload(GPT2)
            target = GPT2.GPT2()
            return await target(channel, user)
        except Exception as e:
            return str(e)

    return reload_run

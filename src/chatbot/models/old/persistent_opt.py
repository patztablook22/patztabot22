_opt_1_3b = None

def get_opt1_3b():
    global _opt_1_3b

    if _opt_1_3b is None:
        from transformers import pipeline
        pipeline = pipeline('text-generation', 
                            model='facebook/opt-1.3b',
                            max_new_tokens=20,
                            min_new_tokens=2)
        _opt_1_3b = pipeline

    return _opt_1_3b


class FinetunedGpt():
    def __init__(self, path):
        pass
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print(path)
        print('tokenizer...')
        self._tokenizer = GPT2Tokenizer.from_pretrained(path)
        print('tokenizer done; model...')
        self._model = GPT2LMHeadModel.from_pretrained(path)
        print('model done')


    def predict(self, prompts):
        input_ids = self._tokenizer(prompts, return_tensors='pt', padding=True).input_ids
        output_ids = self._model.generate(input_ids, max_length=50)
        responses = [self._tokenizer.decode(a) for a in output_ids]
        return responses

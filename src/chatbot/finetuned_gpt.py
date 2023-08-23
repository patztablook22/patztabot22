class FinetunedGpt():
    def __init__(self, path):
        pass
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self._tokenizer = GPT2Tokenizer.from_pretrained(path)
        self._model = GPT2LMHeadModel.from_pretrained(path)

    def predict(self, prompts):
        input_ids = self._tokenizer(prompts, return_tensors='pt', padding=True).input_ids
        output_ids = self._model.generate(input_ids, 
                                          max_new_tokens=30, 
                                          min_new_tokens=10, 
                                          do_sample=True,
                                          top_k=50,
                                          repetition_penalty=1.2)
        generated_ids = [oids[len(iids):] for oids, iids in zip(output_ids, input_ids)]
        responses = [self._tokenizer.decode(a) for a in generated_ids]
        return responses

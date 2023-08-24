class FinetunedGpt():
    def __init__(self, path):
        pass
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self._tokenizer = GPT2Tokenizer.from_pretrained(path)
        self._model = GPT2LMHeadModel.from_pretrained(path)

    def predict(self, prompts):
        mend_token = self._tokenizer.encode("[MEND]")[0]
        inputs = self._tokenizer(prompts, return_tensors='pt', padding=True)
        output_ids = self._model.generate(**inputs,
                                          do_sample=True,
                                          max_new_tokens=128, 
                                          min_new_tokens=4,
                                          temperature=1.1,
                                          top_k=50,
                                          top_p=0.9,
                                          repetition_penalty=1.1,
                                          eos_token_id=mend_token,
                                          pad_token_id=mend_token)
        generated_ids = [oids[len(iids):] for oids, iids in zip(output_ids, inputs.input_ids)]
        responses = [self._tokenizer.decode(a, skip_special_tokens=True) for a in generated_ids]
        return responses

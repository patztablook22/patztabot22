import torch
import dataclasses

def generate(input_ids, tokenizer, model, **kwargs):
    input_ids = input_ids[-500:]
    i = kwargs.get('i', 0)
    length = len(input_ids)
    print('generation start', flush=True)
    output_ids = model.generate(input_ids=torch.tensor([input_ids], dtype=torch.long),
                                num_return_sequences=1,
                                eos_token_id=kwargs.get('eos_token_id', None),
                                pad_token_id=tokenizer.pad_token_id,
                                #min_length = length + 3,
                                max_length = length + 3,
                                do_sample=True,
                                temperature=1.5,
                                )[0]
    print('generation end', flush=True)
    response_ids = output_ids.tolist()[len(input_ids):]
    print(tokenizer.decode(response_ids))
    return [1,2,3]

class FakeModel:
    def generate(self, input_ids, *args, **kwargs):
        return torch.tensor([input_ids.tolist()[0] + [4, 5, 6]], dtype=torch.long)

@dataclasses.dataclass
class FakeTokenizer:
    eos_token_id: int = 12345
    pad_token_id: int = 12345

    def __call__(self, *args, **kwargs):
        @dataclasses.dataclass
        class FakeResponse:
            input_ids: list
        return FakeResponse(input_ids=[1,2,3])


    def decode(self, *args, **kwargs):
        return 'fake decode'

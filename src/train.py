#/usr/bin/env python3

from utils.general import *
from utils import training
import os

def train(args):
    from transformers import TrainingArguments, Trainer

    model, tokenizer = get_model_and_tokenizer(args)
    dataset, collator = get_chat_dataset_and_collator(args, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(data_dir, "training"),
        logging_dir=os.path.join(data_dir, "training", "logs"),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        save_steps=50000,
        per_device_train_batch_size=bsize,
        per_device_eval_batch_size=bsize,
        warmup_steps=10,
        #evaluation_strategy='epoch'
    )


def get_model_and_tokenizer(args):
    #from transformers import LlamaForCausalLM
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    model = LlamaForCausalLM.from_pretrained(args.hub_model_variant)
    tokenizer = GPT2Tokenizer.from_pretrained(args.hub_model_variant)
    model.resize_token_embeddings(len(tokenizer))
    return model

def get_chat_dataset_and_collator(args, tokenizer):
    eos_token_id = 50256
    data_path = 'tokenized.pkl'

    import pickle
    with open(os.path.join(args.data_dir, data_path), 'rb') as f:
        tokenized = pickle.load(f)
    ds = training.MaskedDataset(tokenized, max_tokens=args.max_tokens)
    col = training.DataCollator(pad_token_id=eos_token_id)
    return ds, col

def main(args):
    train(args)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument('--hub-model-variant', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

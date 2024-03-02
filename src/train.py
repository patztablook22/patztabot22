#/usr/bin/env python3

from utils.general import *
from utils import training
import os
import sys
import shellbot

def train(args):
    from transformers import TrainingArguments, Trainer

    shellbot.log("Loading model and tokenizer", ...)
    model, tokenizer = get_model_and_tokenizer(args)
    shellbot.success()

    shellbot.log("Loading dataset and collator", ...)
    dataset, collator = get_chat_dataset_and_collator(args, tokenizer=tokenizer)
    shellbot.success()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        save_strategy='epoch',
        per_device_train_batch_size=args.bsize,
        warmup_steps=args.warmup_steps,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset
    )
    trainer.callback_handler.callbacks = [training.LogCallback()]

    shellbot.log("Starting training")
    trainer.train()
    shellbot.log("Training finished")

    shellbot.log("Saving tokenizer", ...)
    tokenizer.save_pretrained(args.output_dir)
    shellbot.success()

    shellbot.log("Saving model", ...)
    model.save_pretrained(args.output_dir)
    shellbot.success()


def get_model_and_tokenizer(args):

    if args.hf_token_path:
        hf_token = open(args.hf_token_path).read().strip() 
    else:
        hf_token = None

    EXTRA_TOKENS = ['<|types|>', '<|reacts|>']

    def configure(model, tokenizer):
        tokenizer.add_tokens(EXTRA_TOKENS)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def gpt2():
        from transformers import AutoTokenizer, GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(args.model, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        return configure(model, tokenizer)
        
    if args.model.startswith('gpt2'):
        return gpt2()

    raise NotImplementedError(f"{args.model=}")


def get_chat_dataset_and_collator(args, tokenizer):
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length
        if max_tokens is None:
            raise ValueError("couldn't infer `args.max_tokens`")

    import pickle
    with open(args.dataset_path, 'rb') as f:
        tokenized = pickle.load(f)
        ds = training.MaskedDataset(tokenized, max_tokens=max_tokens)

    eos_token_id = tokenizer.eos_token_id
    col = training.DataCollator(pad_token_id=eos_token_id)
    return ds, col

def log_args(args):
    buff = {}
    for key in vars(args):
        buff[str(key)] = getattr(args, key)
    left = max([len(key) for key in buff])
    print("=== ARGS =====================================")
    print("", "".ljust(left), " |")
    for key, value in buff.items():
        print("", key.ljust(left), " |  ", value)
    print("", "".ljust(left), " |")
    print("==============================================")
    sys.stdout.flush()

def main(args):
    log_args(args)
    train(args)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--hf-token-path', type=str)
    parser.add_argument('--max_tokens', type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bsize", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

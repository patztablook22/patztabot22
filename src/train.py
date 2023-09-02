import shellbot

shellbot.log('Importing libraries', ...)

import threading
import torch
import sys, os, time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from dataset import ChatDataset
import numpy as np

shellbot.success()


def train(data_dir):
    train_path = os.path.join(data_dir, 'train.txt')
    val_path = os.path.join(data_dir, 'val.txt')
    model_name = 'gpt2-xl'
    tokenizer_name = model_name
    block_size = lambda i: np.random.choice([64, 128, 256])
    epochs = 8
    bsize = 4
    save_dir = os.path.join(data_dir, "chat_model11")
    whitelist = ["patz", "Patztablook TwentyTwo", "you",
                 "Sběratel Banánů", "Alexander Terziev",
                 "Martin McNickle", "Jan Zasadil", "Filip Kastl",
                 "Jaroslav Žukov", "Robin Stringer", "Jakub Tichanek"]

    
    print(f'{epochs=} {bsize=}')
    print(f'{save_dir=}')
    sys.stdout.flush()

    special_tokens = {
        'conversation_start': '[CSTART]',
        'conversation_end': '[CEND]',
        'message_start': '[MSTART]',
        'breakpoint': '[BREAK]',
        'message_end': '[MEND]',
        'writes': '[WRITES]',
        'avoid': '[AVOID]',
    }

    shellbot.log(f"Creating tokenizer ({tokenizer_name=})", ...)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    shellbot.success()

    print(f"{train_path=}")
    print(f"{val_path=}", flush=True)

    shellbot.log(f"Loading datasets ({block_size=})", ...)
    train_dataset = ChatDataset(
        tokenizer=tokenizer,
        text_path=train_path,
        block_size=block_size,
        special_tokens=special_tokens,
        whitelist=whitelist
    )
    val_dataset = ChatDataset(
        tokenizer=tokenizer,
        text_path=val_path,
        block_size=block_size,
        special_tokens=special_tokens,
        whitelist=whitelist
    )
    shellbot.success()

    shellbot.log("Creating data collator", ...)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    shellbot.success()

    shellbot.log(f"Creating model ({model_name=})", ...)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    shellbot.success()

    shellbot.log("Preparing training args", ...)
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
    shellbot.success()

    shellbot.log("Creating trainer", ...)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        #eval_dataset=val_dataset
    )
    trainer.callback_handler.callbacks = [LogCallback()]
    shellbot.success()

    shellbot.log("Starting training")
    trainer.train()
    shellbot.log("Training finished")

    shellbot.log("Saving tokenizer", ...)
    tokenizer.save_pretrained(save_dir)
    shellbot.success()

    shellbot.log("Saving model", ...)
    model.save_pretrained(save_dir)
    shellbot.success()

class LogCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_step_end(self, args, state, control, **kwargs):
        cur = state.global_step - 1
        end = state.max_steps
        if cur % (end // 100) == 0 or cur == end - 1:
            self._print_progress(cur, end)

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs, flush=True)

    def _print_progress(self, current, total):
        pad = len(str(total))
        shellbot.log(str(current + 1).rjust(pad), '/', total, ...)

class FlushCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        print(flush=True)

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        print(flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        print(flush=True)

    def on_predict(self, args, state, control, **kwargs):
        print(flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        print(flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        print(flush=True)


def main(argv):
    data_dir = 'rp-patrik-zavoral/data'
    train(data_dir)

if __name__ == '__main__':
    main(sys.argv)

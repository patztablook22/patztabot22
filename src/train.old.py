import torch
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import os
from dataset import ChatDataset
import numpy as np

def main(argv):
    data_dir = argv[1]
    with open(os.path.join(data_dir, "train_log.txt"), "w") as log:
        train(data_dir, log)

def train(data_dir, log):

    train_path = os.path.join(data_dir, 'train.txt')
    val_path = os.path.join(data_dir, 'val.txt')
    model_name = os.path.join(data_dir, 'chat_model8')
    tokenizer_name = model_name
    block_size = lambda i: np.random.choice([32, 48, 64, 96, 96, 128, 128, 192, 256, 512])
    block_size = lambda i: np.random.choice([64, 128, 256])
    epochs = 4
    bsize = 4
    save_dir = os.path.join(data_dir, "chat_model9")
    whitelist = ["patz", "Patztablook TwentyTwo", "you",
                 "Sběratel Banánů", "Alexander Terziev",
                 "Martin McNickle", "Jan Zasadil", "Filip Kastl",
                 "Jaroslav Žukov", "Robin Stringer", "Jakub Tichanek"]

    print("train path:", train_path, file=log)
    print("val path:", val_path, file=log)

    special_tokens = {
        'conversation_start': '[CSTART]',
        'conversation_end': '[CEND]',
        'message_start': '[MSTART]',
        'breakpoint': '[BREAK]',
        'message_end': '[MEND]',
        'writes': '[WRITES]',
    }

    print(f"creating tokenizer ({model_name=})... ", end="", file=log)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    print("done", file=log)


    print(f"creating datasets ({block_size=})... ", end="", file=log)
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
    print("done", file=log)

    print("creating data collator... ", end="", file=log)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("done", file=log)

    print(f"creating model ({model_name=})... ", end="", file=log)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    print("done", file=log)

    print("preparing training args... ", end="", file=log)
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
    print("done", file=log)

    print("preparing trainer... ", end="", file=log)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        #eval_dataset=val_dataset
    )
    print("done", file=log)

    # Train the model
    print("starting training", file=log)
    trainer.train()
    print("training finished", file=log)

    print("saving tokenizer... ", end="", file=log)
    tokenizer.save_pretrained(save_dir)
    print("done", file=log)

    print("saving model... ", end="", file=log)
    model.save_pretrained(save_dir)
    print("done", file=log)

if __name__ == '__main__':
    main(sys.argv)

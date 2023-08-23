import torch
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import os

def main(argv):
    data_dir = argv[1]
    with open(os.path.join(data_dir, "train_log.txt"), "w") as log:
        train(data_dir, log)

def train(data_dir, log):
    train_path = os.path.join(data_dir, 'train.txt')
    val_path = os.path.join(data_dir, 'val.txt')
    print("train path:", train_path, file=log)
    print("val path:", val_path, file=log)

    special_tokens = {
        'conversation_start': '[CSTART]',
        'conversation_end': '[CEND]',
        'message_start': '[MSTART]',
        'message_end': '[MEND]',
        'writes': '[WRITES]',
    }

    model_name = 'gpt2-Large'

    print(f"creating tokenizer ({model_name=})... ", end="", file=log)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    print("done", file=log)

    block_size=1024


    print(f"creating datasets ({block_size=})... ", end="", file=log)
    train_dataset = TextDataset(tokenizer=tokenizer,
                                file_path=train_path,
                                block_size=block_size)
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=val_path,
                              block_size=block_size)
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
        overwrite_output_dir=True,
        num_train_epochs=15,
        save_steps=50000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10
    )
    print("done", file=log)

    print("preparing trainer... ", end="", file=log)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    print("done", file=log)

    # Train the model
    print("starting training", file=log)
    trainer.train()
    print("training finished", file=log)

    print("saving tokenizer... ", end="", file=log)
    tokenizer.save_pretrained(os.path.join(data_dir, "chat_model3"))
    print("done", file=log)

    print("saving model... ", end="", file=log)
    model.save_pretrained(os.path.join(data_dir, "chat_model3"))
    print("done", file=log)

if __name__ == '__main__':
    main(sys.argv)

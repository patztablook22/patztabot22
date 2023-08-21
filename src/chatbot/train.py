import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer

train_path = '../data/train.txt'
val_path = '../data/val.txt'

special_tokens = {
    'conversation_start': '[CSTART]',
    'conversation_end': '[CEND]',
    'message_start': '[MSTART]',
    'message_end': '[MEND]',
    'writes': '[WRITES]',
}

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})

block_size=128


train_dataset = TextDataset(tokenizer=tokenizer,
                            file_path=train_path,
                            block_size=block_size)
val_dataset = TextDataset(tokenizer=tokenizer,
                          file_path=val_path,
                          block_size=block_size)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_steps=10,
    save_steps=10,
    warmup_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=val_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# (Optional) Save the model
model.save_pretrained("./chat_model")
tokenizer.save_pretrained("./chat_model")

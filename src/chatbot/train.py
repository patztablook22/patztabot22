import torch
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import os

data_dir = sys.argv[1]

train_path = os.path.join(data_dir, 'train.txt')
val_path = os.path.join(data_dir, 'val.txt')

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

block_size=1024


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
    output_dir=os.path.join(data_dir, "results2"),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_steps=10,
    save_steps=10,
    warmup_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# (Optional) Save the model
model.save_pretrained(os.path.join(data_dir, "chat_model2"))
tokenizer.save_pretrained(os.path.join(data_dir, "chat_model2"))

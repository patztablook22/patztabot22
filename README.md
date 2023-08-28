# RP Patrik Zavoral

out-of-the-box:
- `notebook` - interactive notebooks 
- `src` - training and deployment logic
- `bin` - entry scripts (always use these instead of running `src` files directly)

not a part of the repo:
- `data` - datasets, models, cache, ...
- `config`
  - `discord_token.txt` - secret Discord bot token
  - `discord_config.ini` - default Discord bot permissions etc.

## RP specification

### Personal AI double

The goal is to train a LLM on the author's messenges online (collected from Meta's Messenger) and integrate the model
into a Discord chatbot. 
The motivation was to create an AI double of the author.
The chatbot should therefore (given e.g. the size limitations of the model)
resemble the author's own way of chatting, and generate (again, given the limitations) appropriate
responses using a live prompt pipeline directly from the Discord chatting application. The Discord bot should have some other
features accessible through commands, such as permissions (whitelist or a blacklist for chatting, ...), and status report.

Subtasks for the RP will possibly include:
- creating a Discord bot with a prompt pipeline for collecting message context
- collecting and preprocessing Messenger data into a dataset
- fine tuning a LLM (small enough for individual use) using a remote cluster on the dataset
- integrating the LLM into the Discord bot
- testing the bot manually or automatically on a set of scenarios (introduction, small talk, discussion, ...)

The Bachelor thesis is expected to focus on what is currently called "AI alignment", especially on aligning pretrained LLMs for downstream tasks, e.g. using parameter efficient fine tuning (PEFT) techniaues, prompt tuning, RLHF, etc.


# Status

- Discord bot
    - prompt pipeline feeding the model message history, 
      triggered automatically in DMs or when pinged in public text channels
    - permission hierarchy with slash command integration (`/permissions`)
      (owner, admin, mod, chat, ignored); 
      cached and restored when restarting the bot
    - automatic testing mode triggered by a slash command (`/test`);
      test messages taken from selected text channels 
      (all visible text channels with the prefix `patztabot22-test-`)
    - sped up generated response time from approximately 30s to 3s
      (thanks to custom generation parameters and using separate huggingface 
       APIs for tokenization and generation)
- Models
    - fine-tuned several models (openai-gpt, ..., gpt2-xl) on about 4 years
      of preprocessed messenger data
    - non-zero loss only on whitelisted users (me and a few others)
        - other users only provide generation context 
          without being part of the generated distribution itself
    - experimented with a few prompt and generation modifications, 
      notably stopping conditions and sampling parameters; interestingly
      slight "repetition bonus" (around `repetition_penalty=0.95`) seems to help maintaining conversation flow


# RP Patrik Zavoral

out-of-the-box:
- `notebook` - interactive notebooks 
- `src` - training and deployment logic
- `bin` - entry scripts (always use these instead of running `src` files directly)

not a part of the repo:
- `data` - datasets, models, cache, ...
- `config`
  - `patztabot_token.txt` - secret Discord bot token for patztabot
  - `patztabot_config.ini` - default Discord bot permissions etc. for patztabot
  - `shellbot_token.txt` - secret Discord bot token for shellbot
  - `shellbot_config.ini` - default Discord bot permissions etc. for shellbot

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

- Discord bot (patztabot22)
  - prompt pipeline feeding the model message history, 
    triggered automatically in DMs or when pinged in public text channels
  - permission hierarchy includng server roles, custom thread creation, context window manipulation features (`/permissions`, `/thread`, `/reset`, `!` message prefix)
  - automatic testing mode triggered by a slash command (`/test`);
    test messages taken from selected text channels
  - sped up generated response time from approximately 1 minute to 10 seconds
    (thanks to custom generation parameters and using separate huggingface 
     APIs for tokenization and generation)
- Helper discod bot (Shell)
  - Essentially a Discord logger for training. Metacentrum has very impractical log handling. This bot enables me to see the training program's STDOUT in real time through Discord and to "skip" waiting in the Metacentrum's queue.
- Models
  - fine-tuned several models (openai-gpt, ..., gpt2-xl) on about 5 years
    of preprocessed messenger data and simulated scenarios
  - masked loss for all but whitelisted users (me and a few others)
    - other users only provide generation context 
      without being part of the generated distribution itself
  - custom generation parameters 


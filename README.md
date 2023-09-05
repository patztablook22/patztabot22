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


# Documentation

## patztabot22

The Discord bot `patztabot22` is the focal point of the project. It represents the "AI persona" (an AI double of `patztablook22`) and facilitates all interaction with the underlying LLM.

### Communication

To communicate with `patztabot22`, a simple message has to be sent from a `visible` user (read #permissions) either:
- In private messages,
- In a thread created using the bot's `/thread` command (read #commands),
- In any channel with its name (or its category's name) prefixed `patztabot22-direct`,
- In any public channel when directly pinged.

When communication is triggered, the model enqueues the message context for response generation. When the LLM worker consumes the context, the `patztabot22 is typing...` Discord status is triggered for as long as the model is generating. The bot skips all messages by non-`visible` users.

There are two important features regarding message context collection:
- Using the command `/reset` (read #commands) prevents the model from seeing earlier messages. This is useful e.g. when the model gets confused by a particular message.
- Any message with the prefix `!` (e.g. "! hello world") is completely ignored by the bot. This makes it easy to make hidden notes or communciate with other users in channels where normal messages trigger a response.

### Permissions

The primary permission system of the bot is constituted by the following hierarchy:
- Onwer
- Admin
- Mod
- Visible
- Ignored

The default level assigned to users and specific user permissions (e.g. the owner) can be given to the bot in the config file `config/patztabot_config.ini`:

```ini
[permissions]

# ignored by default (level 0)
default = 0

# the Discord user with given ID is the owner (level 4)
saved =
    620968062044209172 4
```

The primary permissions can be changed by using the permission `/permission` command (see #commands). A user can change the permissions of only lower-level users and can never grant higher or equal level to its own (for example, an admin has access to the bottom three levels - ignored, visible, mod - but not to the owners and to itself). Permissions are cached so that restarting the bot does not necessitate manually setting up the permissions again.

In addition to this global (across-server) hierarchy, the bot can take into account the roles on trusted servers (such as the private development/testing server - `Sanctuary of the great patztabot22`). For example, a user with the role `visible` is automatically visible to the bot without the need of modifying its primary permission hierarchy.


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


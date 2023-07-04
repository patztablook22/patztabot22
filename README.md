# RP Patrik Zavoral

- `notebook` - i usually run jupyter lab here to try stuff
- `src` - code
    - `trafos` - manually implemented transformer components
        - in Torch and [Nite](https://github.com/patztablook22/nite), which is a library (essentially a wrapper)
        I am writing on top of it to \
        a) just play around with Torch \
        b) make the code more expressive and certain tasks easier
        - contains various self-attention components, FFN, encoders, decoders, architectures
        - I migth not end up using any of this at the end at all, but it's still there for me 
          as I want to really know what I am dealing with

## RP specification

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

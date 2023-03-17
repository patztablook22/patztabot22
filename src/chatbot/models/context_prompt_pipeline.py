import discord

async def context_prompt_pipeline(channel: discord.TextChannel,
                                  self_user,
                                  self_nick,
                                  history_messages_limit,
                                  history_length_limit,
                                  prefix,
                                  history_break = None,
                                  prefix_messages = [],
                                  message_format = None,
                                  postfix = None,
                                  ):

    if message_format == None:
        message_format = lambda nick, content: f'{nick} said: "{content}", \n'

    buffer = []
    for nick,content_raw in prefix_messages:
        content = ' '.join(content_raw.split())
        buffer.append(message_format(nick, content))

    buffer1 = []
    length = 0
    async for m in channel.history(limit=history_messages_limit, oldest_first=False): 
        content = ' '.join(m.content.split())

        if content == history_break:
            break

        if m.author == self_user:
            nick = self_nick
        else:
            nick = m.author.name

        s = message_format(nick, content)
        length += len(s)
        if length > history_length_limit:
            break
        buffer1.append(s)

    if not buffer1:
        return None

    if postfix is None:
        postfix = f'{self_nick} said: "'

    prompt = prefix + ''.join(buffer + list(reversed(buffer1))) + postfix
    return prompt

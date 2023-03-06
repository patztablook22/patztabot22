class CommandLogger:
    def __init__(self, interaction = None):
        self.interaction = interaction

    async def write(self, message):
        print(f'> {message}')
        if self.interaction is not None:
            if self.interaction.response.is_done():
                await self.interaction.channel.send(message)
            else:
                await self.interaction.response.send_message(message)

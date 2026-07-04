import asyncio
from aioconsole import ainput
from channels.base import BaseChannel
from bus.message_bus import MessageBus, Message


class CLiChannel(BaseChannel):
    def __init__(self, bus: MessageBus):
        super().__init__("cli")
        self.bus = bus
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._listen())

    async def _listen(self):
        print("Agent CLI 已启动，输入消息（输入 'quit' 退出）")
        while self._running:
            try:
                user_input = await ainput("> ")
                if user_input.lower() == "quit":
                    self._running = False
                    break
                msg = Message(channel_id=self.channel_id, user_id="cli_user", text=user_input)
                await self.bus.publish(msg)
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

    async def stop(self):
        self._running = False

    async def send_message(self, user_id: str, text: str):
        print(f"\n {text}\n")
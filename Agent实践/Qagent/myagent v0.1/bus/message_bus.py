import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Message:
    channel_id: str
    user_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    def __init__(self):
        self.input_queue = asyncio.Queue()

    async def publish(self, message: Message):
        await self.input_queue.put(message)

    async def consume(self) -> Message:
        return await self.input_queue.get()
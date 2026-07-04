import asyncio
from typing import List
from channels.base import BaseChannel
from bus.message_bus import MessageBus
from agent.loop import AgentLoop


class ChannelManager:
    def __init__(self, bus: MessageBus, channels: List[BaseChannel], agent: AgentLoop):
        self.bus = bus
        self.channels = channels
        self.agent = agent

    async def run(self):
        # 启动所有渠道
        tasks = []
        for ch in self.channels:
            tasks.append(asyncio.create_task(ch.start()))

        # 启动消息消费循环
        tasks.append(asyncio.create_task(self._process_messages()))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_messages(self):
        while True:
            msg = await self.bus.consume()
            try:
                reply = await self.agent.handle_message(msg)
                if reply:
                    # 找到对应的渠道发送回复
                    for ch in self.channels:
                        if ch.channel_id == msg.channel_id:
                            await ch.send_message(msg.user_id, reply)
                            break
            except Exception as e:
                print(f"消息处理出错: {e}")
from abc import ABC, abstractmethod


class BaseChannel(ABC):
    def __init__(self, channel_id: str):
        self.channel_id = channel_id

    @abstractmethod
    async def start(self):
        """启动渠道，开始接收消息并推送到 MessageBus"""
        pass

    @abstractmethod
    async def stop(self):
        """停止渠道"""
        pass

    @abstractmethod
    async def send_message(self, user_id: str, text: str):
        """向该渠道的指定用户发送消息"""
        pass
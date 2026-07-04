import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Set

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
)

from channels.base import BaseChannel
from bus.message_bus import MessageBus, Message
from config import settings


class FeishuChannel(BaseChannel):
    def __init__(self, bus: MessageBus):
        super().__init__("feishu")
        self.bus = bus
        self._running = False
        self._ws_thread: threading.Thread | None = None
        self._client: lark.Client | None = None
        self._seen_events: Set[str] = set()
        self._seen_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None   # 主事件循环引用
        self._workspace = Path(settings.workspace_dir)

    async def start(self):
        if not settings.feishu_app_id or not settings.feishu_app_secret:
            print("⚠️ 飞书配置未填写，跳过飞书渠道启动")
            return

        self._client = (
            lark.Client.builder()
            .app_id(settings.feishu_app_id)
            .app_secret(settings.feishu_app_secret)
            .build()
        )

        self._loop = asyncio.get_running_loop()
        self._running = True

        self._run_ws_listener()
        print(f"[Feishu] WebSocket 已启动 (app_id={settings.feishu_app_id[:8]}...)")

    def _run_ws_listener(self):
        """在独立线程中运行 SDK WebSocket"""
        def handle_message(data: P2ImMessageReceiveV1):
            try:
                self._process_and_dispatch(data)
            except Exception as e:
                print(f"[Feishu] 处理事件异常: {e}")

        event_handler = (
            lark.EventDispatcherHandler.builder(
                settings.feishu_verification_token or "",
                settings.feishu_encrypt_key or "",
            )
            .register_p2_im_message_receive_v1(handle_message)
            .build()
        )

        ws_client = lark.ws.Client(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.ERROR,
        )

        import lark_oapi.ws.client as _lark_ws_client

        def _run_ws():
            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            _lark_ws_client.loop = ws_loop
            try:
                while self._running:
                    try:
                        ws_client.start()
                    except Exception:
                        if self._running:
                            time.sleep(5)
            finally:
                ws_loop.close()

        ws_thread = threading.Thread(target=_run_ws, daemon=True)
        ws_thread.start()
        print(f"[Feishu] SDK WebSocket 正在启动...")

    def _process_and_dispatch(self, data: P2ImMessageReceiveV1):
        """解析事件、去重、构造 Message 并通过主事件循环发布到总线"""
        event_id = getattr(data.header, "event_id", None) if hasattr(data, "header") else None
        if event_id:
            with self._seen_lock:
                if event_id in self._seen_events:
                    return
                self._seen_events.add(event_id)
                if len(self._seen_events) > 5000:
                    self._seen_events.clear()

        event = data.event
        msg = event.message
        sender_id = getattr(
            getattr(event.sender, "sender_id", None), "open_id", str(uuid.uuid4())
        )
        chat_id = msg.chat_id or msg.root_id or str(uuid.uuid4())

        text = ""
        images = None
        files = None

        if msg.message_type == "text":
            content = json.loads(msg.content)
            text = content.get("text", "")
        elif msg.message_type == "image":
            content = json.loads(msg.content)
            if content.get("image_key"):
                images = [f"image_key:{content['image_key']}"]
        elif msg.message_type == "file":
            content = json.loads(msg.content)
            file_key = content.get("file_key")
            file_name = content.get("file_name", "unknown")
            if file_key:
                try:
                    self._workspace.mkdir(parents=True, exist_ok=True)
                    safe_name = Path(file_name).name
                    local_path = self._workspace / f"{uuid.uuid4().hex[:8]}_{safe_name}"
                    req = (
                        GetMessageResourceRequest.builder()
                        .message_id(msg.message_id)
                        .file_key(file_key)
                        .type("file")
                        .build()
                    )
                    resp = self._client.im.v1.message_resource.get(req)
                    if resp.success():
                        with open(local_path, "wb") as f:
                            f.write(resp.file.read())
                        files = [str(local_path)]
                        print(f"[Feishu] 文件已下载: {local_path}")
                    else:
                        print(f"[Feishu] 文件下载失败: code={resp.code}")
                except Exception as e:
                    print(f"[Feishu] 文件处理异常: {e}")
        else:
            text = f"[{msg.message_type}]"

        message_obj = Message(
            channel_id=self.channel_id,
            user_id=sender_id,
            text=text,
            metadata={
                "chat_id": chat_id,
                "images": images,
                "files": files,
                "msg_type": msg.message_type,
            },
        )

        asyncio.run_coroutine_threadsafe(
            self.bus.publish(message_obj), self._loop
        )

    async def send_message(self, user_id: str, text: str):
        if not self._client:
            return
        content = json.dumps({"text": text})
        req = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(user_id)
                .msg_type("text")
                .content(content)
                .build()
            )
            .build()
        )

        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None, lambda: self._client.im.v1.message.create(req)
            )
            if not resp.success():
                print(f"[Feishu] 发送失败: code={resp.code}, msg={resp.msg}")
            else:
                print(f"[Feishu] 已回复用户 {user_id}")
        except Exception as e:
            print(f"[Feishu] 发送异常: {e}")

    async def stop(self):
        self._running = False
        print("[Feishu] 已停止")
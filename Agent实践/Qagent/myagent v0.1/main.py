import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio
import click
from config import settings
from llm.factory import create_llm
from tools.registry import tool_registry
from tools.file_tools import register_file_tools
from memory.manager import MemoryManager
from agent.loop import AgentLoop
from channels.manager import ChannelManager
from channels.cli import CLiChannel
from bus.message_bus import MessageBus


# 注意：不再在此处导入 FeishuChannel

@click.group()
def cli():
    pass


@cli.command()
def run():
    """启动所有渠道（CLI + 飞书）"""
    asyncio.run(start_all())


@cli.command()
def run_cli():
    """仅启动命令行渠道"""
    asyncio.run(start_cli_only())


async def start_all():
    # 初始化基础组件（这些是通用的）
    bus = MessageBus()
    llm = create_llm(settings)
    register_file_tools()
    mem_manager = MemoryManager()
    agent = AgentLoop(llm, tool_registry, mem_manager, bus)

    cli_ch = CLiChannel(bus)
    channels = [cli_ch]

    # 尝试导入飞书渠道（如果配置已填写）
    if settings.feishu_app_id and settings.feishu_app_secret:
        try:
            from channels.feishu import FeishuChannel  # 延迟导入
            feishu_ch = FeishuChannel(bus)
            channels.append(feishu_ch)
        except Exception as e:
            print(f"⚠️ 飞书渠道加载失败: {e}")
    else:
        print("ℹ️ 飞书配置未填写，仅启动 CLI 渠道")

    channel_mgr = ChannelManager(bus, channels, agent)
    await channel_mgr.run()


async def start_cli_only():
    bus = MessageBus()
    llm = create_llm(settings)
    register_file_tools()
    mem_manager = MemoryManager()
    agent = AgentLoop(llm, tool_registry, mem_manager, bus)

    cli_ch = CLiChannel(bus)
    channel_mgr = ChannelManager(bus, [cli_ch], agent)
    await channel_mgr.run()

if __name__ == "__main__":
    cli()
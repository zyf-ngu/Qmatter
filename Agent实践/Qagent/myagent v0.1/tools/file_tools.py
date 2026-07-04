import os
from pathlib import Path
from tools.base import Tool
from tools.registry import tool_registry
from config import settings


async def read_file(path: str) -> str:
    """读取文件内容"""
    full_path = Path(settings.workspace_dir) / path
    if not full_path.is_file():
        return f"File '{path}' not found."
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {e}"


async def list_dir(path: str = ".") -> str:
    """列出目录内容"""
    full_path = Path(settings.workspace_dir) / path
    if not full_path.is_dir():
        return f"Directory '{path}' not found."
    try:
        entries = os.listdir(full_path)
        return "\n".join(entries) if entries else "(empty)"
    except Exception as e:
        return f"Failed to list directory: {e}"


def register_file_tools():
    tool_registry.register(Tool(
        name="read_file",
        description="Read the contents of a file in the workspace",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path relative to workspace"}
            },
            "required": ["path"],
        },
        func=read_file
    ))
    tool_registry.register(Tool(
        name="list_dir",
        description="List files and directories in a given workspace directory",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to workspace, default '.'"}
            },
        },
        func=list_dir
    ))
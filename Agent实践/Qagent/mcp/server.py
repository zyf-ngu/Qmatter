from future import annotations

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
ListResourcesRequest,
ListResourcesResult,
ListToolsRequest,
ListToolsResult,
Tool,
)


server = Server("autosar-test-mcp")


@server.list_tools()
async def list_tools(_: ListToolsRequest) -> ListToolsResult:
    return ListToolsResult(
            tools=[
               Tool(
                 name="add",
                 description="Add two integers.",
                 inputSchema={
                    "type": "object",
                    "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                       },
                    "required": ["a", "b"],
                 },
            )
            ]
            )

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "add":
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

    a = int(arguments.get("a"))
    b = int(arguments.get("b"))
    return {"content": [{"type": "text", "text": str(a + b)}]}


@server.list_resources()
async def list_resources(_: ListResourcesRequest) -> ListResourcesResult:
    return ListResourcesResult(
            resources=[
             {
              "uri": "test://hello",
               "name": "hello",
               "description": "A simple greeting resource.",
               "mimeType": "text/plain",
                }
            ]
            )


@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri != "test://hello":
        return ""

    return "hello from autosar-test-mcp"


async def main() -> None:
    # Newer mcp requires explicit initialization options.
    from mcp.server import InitializationOptions
    from mcp.types import ResourcesCapability, ServerCapabilities, ToolsCapability

    init_opts = InitializationOptions(
        server_name="autosar-test-mcp",
        server_version="0.1.0",
        capabilities=ServerCapabilities(
        tools=ToolsCapability(listChanged=False),
        resources=ResourcesCapability(subscribe=False, listChanged=False),
          ),
      )

async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, init_opts)


if __name__== "__main__":
    asyncio.run(main())
from future import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    # Ensure the repo root is on PYTHONPATH for the child process so
    # -m test.mcp_server works even when this script is run by filename.
    repo_root = str(Path(file).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Spawn the server using the same Python interpreter.
    server = StdioServerParameters(
          command=sys.executable,
          args=["-m", "test.mcp_server"],
          env=env,
      )

async with stdio_client(server) as (read_stream, write_stream):
    async with ClientSession(read_stream, write_stream) as session:
       await session.initialize()

tools = await session.list_tools()
print("tools:", [t.name for t in tools.tools])

resources = await session.list_resources()
print("resources:", [r.uri for r in resources.resources])

add_res = await session.call_tool("add", {"a": 2, "b": 3})
# add_res.content is a list of content blocks.
print("add result:", add_res.content[0].text)

rr = await session.read_resource("test://hello")
print("resource text:", rr.contents[0].text)


if __name__ == "__main__":
    asyncio.run(main())
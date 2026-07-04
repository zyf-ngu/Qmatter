import asyncio, sys, json, httpx, websockets

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import settings

async def main():
    # 获取 token
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": settings.feishu_app_id, "app_secret": settings.feishu_app_secret},
            timeout=10
        )
    data = resp.json()
    token = data["tenant_access_token"]
    print("token 获取成功")

    # 测试连接
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with websockets.connect(
            "wss://open.feishu.cn/open-apis/bot/v1/ws",
            additional_headers=headers,
            ping_interval=20,
        ) as ws:
            print("连接成功！等待消息（按 Ctrl+C 停止）")
            async for msg in ws:
                print("收到:", msg[:100])
    except Exception as e:
        print(f"连接失败: {type(e).__name__}: {e}")

asyncio.run(main())
import asyncio, httpx, json
from config import settings

async def main():
    # 1. 获取 token
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": settings.feishu_app_id, "app_secret": settings.feishu_app_secret}
        )
        token = r.json().get("tenant_access_token")
        if not token:
            print("❌ 获取 token 失败:", r.text)
            return
        print("✅ token 获取成功")

    # 2. 获取长连接地址（必须用 POST，空 body）
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://open.feishu.cn/open-apis/event/v1/ws/",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={},          # 核心：空 JSON body
            timeout=10.0,
        )
        print("状态码:", r.status_code)
        print("原始响应:", r.text[:500])

        # 尝试解析 JSON
        try:
            data = r.json()
            print("\n✅ JSON 解析成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ JSON 解析失败: {e}")

asyncio.run(main())
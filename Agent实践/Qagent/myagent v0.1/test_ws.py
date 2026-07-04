import asyncio, httpx, json
from config import settings

async def main():
    # 1. 获取 token
    async with httpx.AsyncClient() as c:
        r = await c.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": settings.feishu_app_id, "app_secret": settings.feishu_app_secret}
        )
        if r.status_code != 200:
            print("获取 token 失败:", r.text)
            return
        token = r.json()["tenant_access_token"]
        print("✅ token 已获取")

    # 2. 获取长连接地址（POST + 空 JSON）
    async with httpx.AsyncClient() as c:
        r = await c.post(
            "https://open.feishu.cn/open-apis/event/v1/ws/",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            data="{}",   # 显式空 JSON 字符串
            timeout=10
        )
    print(f"状态码: {r.status_code}")
    print(f"响应头: {dict(r.headers)}")
    print(f"原始响应内容 (前 500 字符):\n{r.text[:500]}")

    # 尝试解析 JSON
    try:
        data = r.json()
        print("✅ JSON 解析成功:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if data.get("code") == 0 and data.get("data", {}).get("url"):
            print("🎉 成功获取长连接地址！")
    except Exception as e:
        print(f"❌ 非 JSON 响应，无法解析: {e}")

asyncio.run(main())
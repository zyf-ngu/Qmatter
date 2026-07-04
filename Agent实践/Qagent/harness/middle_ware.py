class MiddlewareManager:
    def __init__(self):
        self._middlewares = []

    def register(self, middleware):
        self._middlewares.append(middleware)

    def execute(self, context, core_handler):
        # 构建嵌套的调用链
        def create_chain(index):
            # 从最开始输入的index执行，小于注册的中间件数，则一直调用
            if index < len(self._middlewares):
                middleware = self._middlewares[index]
                # 传入 next 为下一层
                return lambda ctx: middleware(ctx, create_chain(index + 1))  # 直到注册的中间件全部执行完成，返回主流程里的函数
            else:
                return core_handler

        chain = create_chain(0)
        return chain(context)


# 使用示例：
def auth_middleware(ctx, next_handler):
    if not ctx.get("token"):
        return "未授权"
    print("验证通过")
    return next_handler(ctx)


def logging_middleware(ctx, next_handler):
    print(f"请求消息: {ctx['message']}")
    res = next_handler(ctx)
    print(f"响应: {res}")
    return res


def core_agent(ctx):
    return f"回答: {ctx['message']} 的处理结果"


manager = MiddlewareManager()
manager.register(auth_middleware)
manager.register(logging_middleware)
ctx = {"token": "secret", "message": "你好"}
result = manager.execute(ctx, core_agent)
print(result)

class Myhook:
    def __init__(self):
        self._hooks = {
            "on_start": [],
            "on_tool_call": [],
            "on_finish": [],
            "on_error": []
        }

    def register_hook(self, hook_point, callback):
        if hook_point not in self._hooks:
            raise ValueError(f"Invalid hook point: {hook_point}")
        self._hooks[hook_point].append(callback)

    def _execute_hooks(self, hook_point, context):
        for callback in self._hooks[hook_point]:
            callback(context)

    def run(self, task):
        ctx = {"task": task, "steps": []}
        self._execute_hooks("on_start", ctx)

        try:
            # 模拟工具调用过程
            if "天气" in task:
                tool_name = "get_weather"
                tool_args = {"location": task.split("查询")[-1].strip()}
                ctx["current_tool"] = tool_name
                ctx["tool_args"] = tool_args
                self._execute_hooks("on_tool_call", ctx)

                # 模拟工具执行结果
                if tool_args["location"] == "火星":
                    raise ValueError("无法查询火星天气")
                ctx["steps"].append(f"调用工具 {tool_name}，参数 {tool_args}")
                ctx["result"] = f"{tool_args['location']} 天气晴朗，25°C"

            else:
                ctx["result"] = "直接回答：我不需要调用工具"

            self._execute_hooks("on_finish", ctx)

        except Exception as e:
            ctx["error"] = e
            self._execute_hooks("on_error", ctx)
            ctx["result"] = f"执行出错：{e}"

        return ctx.get("result")


# 自定义回调函数示例
def log_start(ctx):
    print(f"[START] 任务：{ctx['task']}")

def log_tool_call(ctx):
    print(f"[TOOL] 调用 {ctx.get('current_tool')}，参数 {ctx.get('tool_args')}")

def log_finish(ctx):
    print(f"[FINISH] 结果：{ctx['result']}")
    if ctx.get("steps"):
        print(f"        执行步骤：{ctx['steps']}")

def log_error(ctx):
    print(f"[ERROR] {ctx['error']}")

# 额外演示：多个钩子绑定到同一事件
def extra_start_log(ctx):
    print(f"[EXTRA] 开始时间标记，任务长度：{len(ctx['task'])}")


# 运行演示
if __name__ == "__main__":
    # 创建钩子实例并注册各种回调
    myhook = Myhook()
    myhook.register_hook("on_start", log_start)
    myhook.register_hook("on_start", extra_start_log)   # 同一钩子点多个回调
    myhook.register_hook("on_tool_call", log_tool_call)
    myhook.register_hook("on_finish", log_finish)
    myhook.register_hook("on_error", log_error)

    print("=== 正常任务：查询天气 ===")
    result = myhook.run("查询天气 北京")
    print(f"最终返回：{result}\n")

    print("=== 错误任务：查询火星天气（触发 on_error）===")
    result = myhook.run("查询天气 火星")
    print(f"最终返回：{result}\n")

    print("=== 无工具调用任务 ===")
    result = myhook.run("你好，介绍一下自己")
    print(f"最终返回：{result}")
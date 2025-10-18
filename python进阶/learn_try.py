
# # try except 的基本用法
# try:
#     x = 10 # 定义在 try 块中
#     y = 5 / 0 # 这里会引发异常
# except ZeroDivisionError:
#     print(x)  # 可以访问 x（因为异常发生在赋值之后）
#     # print(y) # 错误！y 未定义（因为赋值未完成）
#     z = 20  # 在except 块中定义新变量
# print(z)  # 可以访问 z（值为 20）



# traceback模块
import traceback

def function_a():
    return function_b()


def function_b():
    return function_c()


def function_c():
    # 这里会引发一个错误
    return 1 / 0  # 除零错误

# 普通异常处理 - 信息有限
try:
    function_a()
except Exception as e:
    print("普通异常处理:")
    print(f"异常类型: {type(e).__name__}")
    print(f"异常信息: {e}")
    print("-" * 50)
#
# 使用 traceback 获取详细信息
try:
    function_a()
except Exception as e:
    print("使用 traceback.print_exc 处理:")
    print("完整异常信息:")
    # traceback.print_exc()  # 打印完整的异常跟踪信息
    print("-" * 50)
    # 获取异常信息作为字符串
    error_info = traceback.format_exc()
    print("使用traceback.print_exc输出 :")
    print(error_info)
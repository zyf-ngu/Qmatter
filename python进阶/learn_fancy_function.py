# # 内部函数——闭包
# def make_counter():
#     count = 0  # 外部函数的变量
#
#     def counter():
#         nonlocal count
#         count += 1
#         return count
#
#     # 返回内部函数（闭包）
#     return counter
#
#
# # 创建计数器实例
# counter1 = make_counter()
# counter2 = make_counter()
# # 调用闭包，每个实例会记住自己的count状态
# print(counter1())  # 1
# print(counter1())  # 2
# print(counter2())  # 1
# print(counter1())  # 3
# print(counter2())  # 2
#
#
# # 实例2：修改外部可变类型变量（list / dict）
# def outer_function():
#     # 外部函数的可变类型变量（列表）
#     outer_list = [1, 2, 3]
#
#     def inner_function():
#         # 直接修改外部可变类型变量（无需特殊声明）
#         outer_list.append(4)
#         print(f"内部函数修改后的列表: {outer_list}")
#
#     inner_function()
#     return outer_list
#
#
# # 调用外部函数
# result = outer_function()
# print(f"外部函数返回的列表: {result}")
#
# # 实例3：修改外部不可变类型变量（需用nonlocal）
#
# def outer_function():
#     # 外部函数的不可变类型变量（整数）
#     outer_num = 10
#
#     def inner_function():
#         # 声明需要修改外部不可变变量
#         nonlocal outer_num
#         outer_num = 20  # 现在可以修改了
#         print(f"内部函数修改后的数值: {outer_num}")
#
#     inner_function()
#     return outer_num
#
# # 调用外部函数
# result = outer_function()
# print(f"外部函数返回的数值: {result}")
# #
#
# # 装饰器函数，接收函数为参数，返回一个扩展的函数

# def decorator_log(func):
#     def wrapper(*args ,**kwargs):
#         print('在原函数执行前扩展一些功能')
#         result =func(*args ,**kwargs)
#         print('在原函数执行后扩展一些功能')
#         return result
#
#     return wrapper
#
#
# @decorator_log
# def add_num(a ,b):
#     print('执行接收函数')
#     return a+ b
#
#
# result = add_num(3, 5)
# print('result', result)
#
#
# # 带参数的装饰器函数
# def require_role(role):
#     # 外层接收装饰器参数
#     def decorator(original_func):
#         # 中间层接收函数
#         def wrapper(*args, **kwargs):
# # 内层实现装饰逻辑
#             if role == "admin":
#                 print("管理员权限验证通过")
#                 return original_func(*args, **kwargs)
#             else:
#                 raise PermissionError("权限不足!")
#         return wrapper
#     return decorator
# @require_role("admin")
# def delete_database():
#     print("数据库已删除!")
#
# @require_role("user")
# def view_data():
#     print("显示数据...")
# # 测试
# delete_database()
# # view_data()
#
# try:
#     view_data()     # 触发异常
# except PermissionError as e:
#     print(f"错误: {e}")

#
# 匿名函数和高阶函数
# 1. lambda基本用法
add = lambda x, y: x + y
print('lambda函数输出1',add(5, 3))  # 输出: 8

# 2. 在排序中使用
students = [("Alice", 90), ("Bob", 85), ("Charlie", 95)]# 按分数排序
sorted_students = sorted(students, key=lambda x: x[1])
print(sorted_students)  # 输出: [('Bob', 85), ('Alice', 90), ('Charlie', 95)]
# 3. 立即调用 lambda 函数
result = (lambda x, y: x * y)(5, 6)
print(result)  # 输出: 30

from functools import reduce

numbers = [1, 2, 3, 4, 5]
# map() - 对每个元素应用函数
squared = list(map(lambda x: x**2, numbers))
print(squared)  # 输出: [1, 4, 9, 16, 25]
# filter() - 过滤满足条件的元素
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # 输出: [2, 4]
# reduce() - 累积计算序列中的元素
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 输出: 15
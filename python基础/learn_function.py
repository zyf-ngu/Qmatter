# # 定义函数
# def calculate_rectangle_area(height, width):
#     return height * width
#
#
# height1 = 6
# width1 = 5
# area1 = calculate_rectangle_area(height=height1, width=width1)
# print('area1:', area1)

#
# # # 位置参数
# def num_add(a, b):
#     return a + b
#
#
# sum1 = num_add(3, 5)
# print('sum1:', sum1)
#
#
# # 关键字参数
# def describe_person(name=None, age=None, city=None):
#     print(f"{name} 今年 {age} 岁，住在 {city}。")
#
#
# describe_person(age=25, city="New York", name="Alice")  # 调用顺序可以不同
# describe_person("Alice", age=25, city="New York")  # 位置参数在前
#
#
# *args不定数量位置参数
# def sum_numbers(*args):
#     total = 0
#     for num in args:
#         total += num
#     return total
#
#
# # 调用时可以传递不同数量的位置实参
# result1 = sum_numbers(1, 2, 3)
# result2 = sum_numbers(10, 20, 30, 40)
# result3 = sum_numbers(5)
# print("传入 1, 2, 3 的求和结果:", result1)
# print("传入 10, 20, 30, 40 的求和结果:", result2)
# print("传入 5 的求和结果:", result3)
#
#
# # 不定数量关键字参数
# def print_info(**kwargs):
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")
#
#
# # 调用函数并传入不同的关键字参数
# print_info(name="Alice", age=25, city="New York")
# print_info(product="Laptop", brand="Dell")

#
# 混合参数实例
# def complex_function(a, b=2, *args, c, d=10, **kwargs):
#     print(f"a = {a}")
#     print(f"b = {b}")
#     print(f"args = {args}")
#     print(f"c = {c}")
#     print(f"d = {d}")
#     print(f"kwargs = {kwargs}")
#
#
# # 调用示例
# complex_function(1, 3, 4, 5, c=6, d=20, e='extra', f=42)
#
#
# # 传递不可变对象（不会影响原值）
# def unchange_obj(num, str1):
#     num = 10  # 重新赋值，创建了一个新的整数对象
#     str1 = 'new-string'
#     print("Inside function:", num, str1)
#
#
# x = 5
# orig_str = 'abc'
# unchange_obj(x, orig_str)
# print("Outside function:", x, orig_str)
#
#
# # 输出: 5,abc
#
#
# # 传递可变对象（增删改会影响原值）
# def change_list(my_list):
#     my_list.append(4)  # 修改列表，添加一个元素
#     print("Inside function:", my_list)
#
#
# lst = [1, 2, 3]
# change_list(lst)
# print("Outside function:", lst)
#
#
# # 输出: [1, 2, 3, 4]
#
# # 传递可变对象但重新赋值（不影响原值）
# def reassign_list(my_list):
#     my_list = [4, 5, 6]  # 重新赋值，指向新的列表
#     print("Inside function:", my_list)
#
#
# lst = [1, 2, 3]
# reassign_list(lst)
# print("Outside function:", lst)
#
#
# # 输出: [1, 2, 3]
#
#
# # 默认值为空时，且要传递可变对象
# # 不好的做法
# def bad_append_to(element, target=[]):
#     target.append(element)
#     return target
#
#
# # 好的做法，默认值为None，在函数内部增加一个判断条件，若为none，则新建一个空对象
# def good_append_to(element, target=None):
#     if target is None:
#         target = []
#     target.append(element)
#     return target
#
#
# bad_target1 = bad_append_to(1)
# print('bad_target1:', bad_target1)  # [1]
# bad_target2 = bad_append_to(2)
# print('bad_target2:', bad_target2)  # [1,2]
#
# good_target1 = good_append_to(1)
# print('good_target1:', good_target1)  # [1]
# good_target2 = good_append_to(2)
# print('good_target2:', good_target2)  # [2]

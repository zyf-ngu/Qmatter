# copy函数
# import copy
# # 原始列表
# original = [1, 2, [3, 4]]
# print("原始列表:", original)
# # 浅拷贝
# shallow = copy.copy(original)
# shallow[0] = 10  # 修改第一层元素
# shallow[2][0] = 30  # 修改第二层元素
# print("浅拷贝后修改:")
# print("原始列表:", original)  # [1, 2, [30, 4]] - 内部子对象列表被修改，数值1不变
# print("浅拷贝列表:", shallow)  # [10, 2, [30, 4]]-
# original = [1, 2, [3, 4]]# 重置
# # 深拷贝
# deep = copy.deepcopy(original)
# deep[0] = 10  # 修改第一层元素
# deep[2][0] = 30  # 修改第二层元素
# print("\n深拷贝后修改:")
# print("原始列表:", original)  # [1, 2, [3, 4]] - 完全不受影响
# print("深拷贝列表:", deep)  # [10, 2, [30, 4]]
#
#
# # 海象运算符

# # 正确：作为独立语句
# length = len([1,2,3])
# # # 错误：不能直接放在表达式中
# if length = len([1,2,3]) > 0:  # SyntaxError: invalid syntax
#     print("非空")
#
#
# # 正确：嵌入到条件表达式中
# if (length := len([1,2,3])) > 0:
#     print(f"长度为：{length}")  # 输出：长度为：3
#
#
# class User:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
# # 创建对象
# user = User("Alice", 30)
# # 获取存在的属性
# name= getattr(user, "name") # 等效于 user.name
# print(f"Name: {name}") # 输出: Name: Alice
# # 获取不存在的属性（提供默认值）
# gender= getattr(user, "gender", "Unknown")
# print(f"Gender: {gender}") # 输出: Gender: Unknown
# # 获取不存在的属性（无默认值）
# try:
#     getattr(user, "address") # 触发 AttributeError
# except AttributeError as e:
#     print(f"Error: {e}") # 输出: Error: 'User' object has no attribute 'address'
#
#
# class Config:
#     pass
# # 创建空对象
# config = Config()
# # 设置新属性
# setattr(config, "timeout", 10) # 等效于 config.timeout = 10
# print(f"Timeout: {config.timeout}") # 输出: Timeout: 10
# # 修改现有属性
# setattr(config, "timeout", 20)
# print(f"Updated Timeout: {config.timeout}") # 输出: Updated Timeout: 20
# # 动态添加方法
# def log_message(self, msg):
#     return f"LOG: {msg}"
# setattr(Config, "log", log_message) # 添加到类（所有实例共享）
# print(config.log("Test")) # 输出: LOG: Test
#
#
# instance
list1=[1,2,3,4]
print(isinstance(list1,list))

# # zip函数
# # 示例 1：列表长度为偶数
# _splits = [10, 20, 30, 40, 50, 60]
# # 切片结果：
# #   _splits[0::2] = [10, 30, 50]  ← 偶数索引
# #   _splits[1::2] = [20, 40, 60]  ← 奇数索引
# result = list(zip(_splits[0::2], _splits[1::2]))
# print(result)  # 输出: [(10, 20), (30, 40), (50, 60)]
# # 示例 2：列表长度为奇数（多余元素被忽略）
# _splits = [10, 20, 30, 40, 50]
# # 切片结果：
# #   _splits[0::2] = [10, 30, 50]  ← 3个元素
# #   _splits[1::2] = [20, 40]      ← 2个元素
#
# result = list(zip(_splits[0::2], _splits[1::2]))
# print(result)  # 输出: [(10, 20), (30, 40)] （50 被忽略）
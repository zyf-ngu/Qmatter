import copy
# 原始列表
original = [1, 2, [3, 4]]
print("原始列表:", original)
# 浅拷贝
shallow = copy.copy(original)
print('shallow',shallow)
shallow[0] = 10  # 修改第一层元素
shallow[2][0] = 30  # 修改第二层元素
print("浅拷贝后修改:")
print("原始列表:", original)  # [1, 2, [30, 4]] - 内部子对象列表被修改，数值1不变
print("浅拷贝列表:", shallow)  # [10, 2, [30, 4]]-
original = [1, 2, [3, 4]]# 重置
# 深拷贝
deep = copy.deepcopy(original)
print('deep',deep)
deep[0] = 10  # 修改第一层元素
deep[2][0] = 30  # 修改第二层元素
print("\n深拷贝后修改:")
print("原始列表:", original)  # [1, 2, [3, 4]] - 完全不受影响
print("深拷贝列表:", deep)  # [10, 2, [30, 4]]
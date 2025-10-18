import torch

# # 基本数学运算
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# # 加法
# add_result = a + b
# print("加法结果:", add_result)
# # 减法
# sub_result = a - b
# print("减法结果:", sub_result)
# # 乘法
# mul_result = a * b
# print("乘法结果:", mul_result)
# # 除法
# div_result = a / b
# print("除法结果:", div_result)
#
#
# # 统计运算 sum，mean，max，min
# tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# # 全部求和
# sum_all=torch.sum(tensor)
# print("按行求和结果:", sum_all)
# # 按行求和（指定维度 0）
# sum_row = torch.sum(tensor, dim=0)
# print("按行求和结果:", sum_row.shape,sum_row)
# # 按列求和（指定维度 1）
# sum_col = torch.sum(tensor, dim=1)
# print("按列求和结果:", sum_col)
# # 按行求和并保持维度
# sum_row_keepdim = torch.sum(tensor, dim=0, keepdim=True)
# print("按行求和并保持维度结果:", sum_row_keepdim.shape,sum_row_keepdim)
#
# #
# # 创建一个示例张量
# tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# # 按行找最大值（指定维度 0）
# max_row, max_row_indices = torch.max(tensor, dim=0)
# print("按行找最大值结果:", max_row)
# print("按行找最大值对应的索引:", max_row_indices)
# # 按列找最大值（指定维度 1）
# max_col, max_col_indices = torch.max(tensor, dim=1)
# print("按列找最大值结果:", max_col)
# print("按列找最大值对应的索引:", max_col_indices)
# # 按行找最大值并保持维度
# max_row_keepdim, _ = torch.max(tensor, dim=0, keepdim=True)
# print("按行找最大值并保持维度结果:", max_row_keepdim)

#
# # tensor幂运算**  使用 ** 运算符进行逐元素的幂运算。
# import torch
# # 创建一个示例张量
# tensor = torch.tensor([1, 2, 3])
# # 幂运算
# power_result = tensor ** 2
# print("幂运算结果:", power_result)
# # 三角函数运算 torch.sin、torch.cos、torch.tan 。主元素运算
#
# import torch
# # 创建一个示例张量
# angle_tensor = torch.tensor([0, torch.pi / 2, torch.pi])
# # 正弦函数运算
# sin_result = torch.sin(angle_tensor)
# print("正弦函数运算结果:", sin_result)
# # 余弦函数运算
# cos_result = torch.cos(angle_tensor)
# print("余弦函数运算结果:", cos_result)



# 矩阵乘法torch.matmul（t1,t2） 或 @


#
# a = torch.randn(2, 3)  # 形状 (2, 3)
# b = torch.randn(3, 4)  # 形状 (3, 4)
# c = torch.matmul(a, b)  # 结果形状 (2, 4)
# print(c.shape)

# # 一维张量 × 一维张量（点积）
#
# a = torch.randn(3)    # 形状 (3,)
# b = torch.randn(3)    # 形状 (3,)
# c = torch.matmul(a, b)  # 结果形状 标量，等价于 a·b
# print(c.shape)
#
# #  一维张量 × 二维张量
# # 规则：一维张量被视为行向量（形状扩展为 (1, n)），与二维矩阵相乘后再挤压掉多余维度。
# a = torch.randn(3)    # 形状 (3,) → 视为 (1, 3)
# b = torch.randn(3, 4)  # 形状 (3, 4)
# c = torch.matmul(a, b)  # 结果形状 (4,)，等价于 (1,3)×(3,4)=(1,4) → 挤压为 (4,)
# print(c.shape)
#
# # 二维张量 × 一维张量
# # 规则：一维张量被视为列向量（形状扩展为 (n, 1)），相乘后挤压维度。
# a = torch.randn(2, 3)  # 形状 (2, 3)
# b = torch.randn(3)    # 形状 (3,) → 视为 (3, 1)
# c = torch.matmul(a, b)  # 结果形状 (2,)，等价于 (2,3)×(3,1)=(2,1) → 挤压为 (2,)
# print(c.shape)
#
# # 5. 高维张量 × 高维张量（批量矩阵乘法）
# # 批次维度形状完全相同
# a=torch.randn(2,5,3,4)   # 前序批次维度 (2,5)，最后两维 (3,4)
# b=torch.randn(2,5,4,6)   # 前序批次维度 (2,5)，最后两维 (4,6)
# c=torch.matmul(a,b)      # 结果形状 (2,5,3,6)# 逻辑：对每个 (2,5) 批次，执行 (3,4)×(4,6)=(3,6) 的矩阵乘法
# print(c.shape)
#
# # 批次维度支持广播（某一维度为 1 时可扩展）
# a = torch.randn(2, 1, 3, 4)  # 批次维度 (2,1)
# b = torch.randn(1, 5, 4, 6)  # 批次维度 (1,5)
# c = torch.matmul(a, b)        # 结果形状 (2,5,3,6)# 逻辑：批次维度先广播为 (2,5)，再对每个批次执行 (3,4)×(4,6)=(3,6)
# print(c.shape)
#
# a = torch.randn(2,  3, 4)  # 批次维度 (2,)
# b = torch.randn(2, 4, 6)  # 批次维度 (2)
# c = torch.bmm(a, b)        # 结果形状 (2,3,6)#
#
# #


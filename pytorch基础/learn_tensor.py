import torch
import numpy as np
import math

# # 属性
# tensor1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
# print('tensor形状', tensor1.shape)
# print('tensor维度', tensor1.ndim)
# print('tensor维度', tensor1.dim())
# print('tensor数据类型', tensor1.dtype)
# # print('tensor是否自动求导',tensor1.auto_grad)

# 张量创建
# # 转化法
# #
# list1 = [[1, 2], [3, 4], [5, 6]]
# tensor1 = torch.tensor(list1)
# print('tensor1:', tensor1)
#
# np1 = np.array([[2, 3], [6, 2], [9, 7]])
# tensor2 = torch.tensor(np1)
# print('tensor2:', tensor2)
#
# arr=np.array([[1, 2, 3], [4, 5, 6]])
# t=torch.from_numpy(arr)
# t[0, 0] =-1
# print('original np:',arr)
# arr[0, 0] = 0
# print('after t:',t)

#
# # # api法
# tensor3 = torch.zeros(2, 3)
# print('tensor3:', tensor3)
#
# tensor4 = torch.ones(2, 3)
# print('tensor4:', tensor4)
#
# tensor5 = torch.eye(2, 3)
# print('tensor5:', tensor5)
#
# tensor6 = torch.full((2, 3), 2)
# print('tensor6', tensor6)
#
# tensor7 = torch.zeros_like(tensor6)
# print('tensor7', tensor7)
#
# tensor8 = torch.full_like(tensor6, 3)
# print('tensor8', tensor8)
#
# tensor9 = torch.arange(0, 10, 3)
# print('tensor9', tensor9)
#
# tensor10 = torch.linspace(0, 10, 6)
# print('tensor10', tensor10)
# # print(math.log(tensor10))
#
# tensor11=torch.logspace(0,2,5)
# print('tensor11', tensor11)

#
# # #
# mean1 = torch.arange(1, 5, dtype=torch.float)
# mean2 = 2.0
# std1 = torch.ones(3, 1, dtype=torch.float)
# std2 = 0.3
# tensor_normal1 = torch.normal(mean1, std1)
# print('tensor_normal1', tensor_normal1)
# tensor_normal2 = torch.normal(mean1, std2)
# print('tensor_normal2', tensor_normal2)
# tensor_normal3 = torch.normal(mean2, std1)
# print('tensor_normal3', tensor_normal3)
# tensor_normal4 = torch.normal(mean2, std2, size=(2, 3))
# print('tensor_normal4', tensor_normal4)
#

#
# # 获取
#
# # 切片获取
# # mean1 = torch.arange(1, 5, dtype=torch.float)
# # std1 = torch.ones(3, 1, dtype=torch.float)
# tensor_normal1 = torch.normal(1, 0.5,size=(3,5))
# # tensor_split = tensor_normal1[0:2, :]
# print(tensor_normal1)
# print('tensor_split', tensor_split)
#
# 索引获取
# index = torch.tensor([0, 1])
# t_select = tensor_normal1.index_select(dim=0, index=index)
# print('t_select', t_select)
#
# mask = tensor_normal1.le(1.0)
# t_mask = torch.masked_select(tensor_normal1, mask)
# print('t_mask', t_mask)
# #
# 张量形状的变化
# # 切分
# t_chunk = torch.chunk(tensor_normal1, chunks=2, dim=0)
# print('t_chunk', t_chunk)
# print(type(t_chunk))
#
# t_split = torch.split(tensor_normal1, [1, 2], dim=0)
# print('t_split', t_split)
# # #
# # # 张量拼接
# t_cat = torch.cat([t_split[0], t_split[1]], dim=0)
# print('t_cat', t_cat)
# print(t_cat.shape)
# print(t_cat == tensor_normal1)
#
# t_stack = torch.stack([t_split[0], t_split[0]], dim=0)
# print('t_stack', t_stack)
# print(t_stack.shape)

# 维度变换
tensor_normal2=torch.normal(1,2,size=(3,4))
t_reshape = torch.reshape(tensor_normal2, (-1, 2, 2))  # -1
# print('t_reshape', t_reshape)
print(t_reshape.shape)

t_transpose = torch.transpose(t_reshape, dim0=0, dim1=1)
# print('t_transpose', t_transpose)
print(t_transpose.shape)


#
# 维度压缩扩充 squeeze和unsqueeze
import torch

t = torch.randn(2, 3, 4)  # 原始形状：(2, 3, 4)

print(t.unsqueeze(0).shape)   # torch.Size([1, 2, 3, 4])
print(t.unsqueeze(1).shape)   # torch.Size([2, 1, 3, 4])
print(t.unsqueeze(2).shape)   # torch.Size([2, 3, 1, 4])
print(t.unsqueeze(3).shape)   # torch.Size([2, 3, 4, 1])
print(t.unsqueeze(-1).shape)  # torch.Size([2, 3, 4, 1]) （等价于 dim=3）
print(t.unsqueeze(-4).shape)  # torch.Size([1, 2, 3, 4]) （等价于 dim=0）


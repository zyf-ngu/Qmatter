# # 字符串
# # 普通创建
# str1='123@string'
# print('str1:',str1)
#
# # 转化
# a=1
# str_trans=str(a)
# print(type(str_trans))
#
#
#
#
# # 添加
# str1='hello,world!'
# str2=str1+'abc'
# print('str2:',str2)
# print('添加前字符串id:',id(str1))
# print('添加后字符串id:',id(str2))
# print(f"两个字符串是同一个对象吗? {str2 is str1}")  # 输出: False
#
#
# # join方法
# parts = []
# for i in range(10):
#     parts.append(str(i))
# str_join = ",".join(parts)  # str.join(sequence)
# print('str_join:',str_join)
# #
# #
# # join方法
# # 使用 join() 方法拼接（效率高）
# import time
# start_time = time.time()
# parts = []
# for i in range(10000):
#     parts.append(str(i))
# str_join = ",".join(parts)  # str.join(sequence)
# end_time = time.time()
# print(f"使用 join() 方法耗时: {end_time - start_time:.4f} 秒")
#
# # 使用 + 操作符拼接（效率低）
# start_time = time.time()
# result = ""
# for i in range(10000):
#     result += str(i)
# end_time = time.time()
# print(f"使用 + 操作符耗时: {end_time - start_time:.4f} 秒")
#
#
#
#
# # 删除
# str3='hello,world!'
# str_replace=str3.replace(',','')
# print('str_replace',str_replace)
#
# str4= "   Hello, World!   "
# str_strip = str4.strip()
# print(f"原始字符串: '{str4}'")
# print(f"处理后字符串: '{str_strip}'")
# print(f"原始字符串是否改变: {str4 is str_strip}")  # 输出: False
#
# # 移除特定字符
# str5 = "***Hello, World!***"
# result = str5.strip('*')
# print(result)  # 输出: Hello, World!
#
# #
# # 查 获取单个元素
# str6='hello,world!'
# str_single=str6[6]
# print('str_single:',str_single)
#
# # 获取多个元素切片
# str7='hello,world!'
# str_split=str7[0:5]
# print(f'str_split:{str_split}')
#
# # 遍历
# str8='hello,world!'
# for str in str8:
#     print(str)
#
#
# 修改
# 字符串不能直接使用下标更改特定值，只能使用切片拼接的方法。
# str9='hello,world!'
# # 尝试修改字符串中的字符会引发错误
# try:
#     str9[0] = "H"  # 这会引发 TypeError
# except TypeError as e:
#     print(f"错误: {e}")
#
# # 正确的方法是创建新字符串#
# # 方法1: 使用切片和连接
# new_text = "H" + str9[1:]
# print(f"方法1结果: {new_text}")
#
# # 方法2: 转换为列表再转回字符串
# text_list = list(str9)
# text_list[0] = "H"
# new_text = "".join(text_list)
# print(f"方法2结果: {new_text}")
#
# #
# # 验证 upper() 函数创建新对象
# original = "hello world"
# print(f"原始字符串: {original}")
# print(f"原始字符串 ID: {id(original)}")
# #
# uppered = original.upper()
# print(f"转换后字符串: {uppered}")
# print(f"转换后字符串 ID: {id(uppered)}")
# print(f"两个字符串是同一个对象吗? {original is uppered}")  # 输出: False
# #
# # 其他转换函数示例
# print("lower():", original.lower(), id(original.lower()))
# print("title():", original.title(), id(original.title()))
#
#
# # 求长度
# str10='hello,world!'
# str_size=len(str10)
# print('str_size:',str_size)


# # 列表
# #
# # 创建
# list1=[]
# list2=[1,'a',list1]
# print('list1:',list1)
# print('list2:',list2)
#
# # 转化
# str1='abc'
# list3=list(str1)
# print('list3:',list3)
#
# # 解析创建
# squares=[value**2 for value in range(1,5)]
# print('squares:',squares)
#
# squares2=[]
# for value in range(1,5):
#     squares2.append(value**2)
# print('squares2:',squares2)


#
# # 添加
# list4=[]
# list4.append(1)
# print('list4:',list4)
#
# list4.insert(0,2)
# print('list4:',list4)

# # 删除
# list5=[1,'a',1,'b',2]
# list_pop=list5.pop(1)
# print('list_pop:',list_pop)
# print('list5:',list5)
#
# list_remove=list5.remove(1)
# print('list_remove:',list_remove)
# print('list5:',list5)
#
# del list5[0]
# print('list5:',list5)

# # 获取
# list6=[1,'a',1,'b',2]
# temp=list6[0]
# print('temp:',temp)
#
# # 切片
# list_split=list6[0:2]
# print('list_split:',list_split)
#
# # 遍历
# for list in list6:
#     print('list:',list)

# # 修改
# list7=[1,'a',1,'b',2]
# list7[0]=10
# print('list7:',list7)
#
# # 求长
# list8=[1,'a',1,'b',2]
# list_len=len(list8)
# print('list_len:',list_len)
# #


# # 元组
#
# # 创建
# tuple1=(1,'abc',[1,2])
# print('tuple1:',tuple1)


# # 不可修改 ×
# tuple1[0]=10
# print('tuple1:',tuple1)
#
# # 获取
# temp=tuple1[1]
# print('temp:',temp)
#
# # 切片
# tuple_split=tuple1[:2]
# print('tuple_split',tuple_split)
#
# # 遍历
# for tup in tuple1:
#     print('tup:',tup)
#
# # 重新定义元组变量
# tuple3=(1,2,3)
# tuple3=(4,5,6)
# print('tuple3:',tuple3)


# 字典

# 创建
# dict1={'abc':'value1',1:[1,2],(1,2):{ }}
# print('dict1:',dict1)
#
# # 添加
# dict1['key']='value2'
# print('dict1:',dict1)
#
# # 删除
# dict2={'abc':'value1',1:[3,4],(5,6):{ }}
# del1=dict2.pop(1,None)
# print('del1:',del1)
# del2=dict2.popitem()
# print('del2:',del2)
# # # 键不存在时，报KeyError: 2
# del3=dict2.pop(2,None)
# #
#
# # # # 获取
# dict3={'abc':'value1',1:[3,4],(5,6):{ }}
# temp=dict3[(5,6)]
# print('temp:',temp)
#
# # 遍历
# for k,v in dict3.items():
#     print('k:',k,'v:',v)
# for k in dict3.keys():
#     print('k:',k)
# for v in dict3.values():
#     print('v:',v)
#
# # 修改
dict3={'abc':'value1',1:[3,4],(5,6):{ }}
dict3[(5,6)]='a'
print('dict3:',dict3)

# update基本使用，参数为另一个字典
dict4 = {'name': 'Alice', 'age': 25}
dict5 = {'age': 26, 'city': 'New York'}

dict4.update(dict5)
print("更新后的字典:", dict4)  # {'name': 'Alice', 'age': 26, 'city': 'New York'}  #键存在，更新值；不存在，添加键值对；

# 使用关键字参数
dict6 = {'a': 1, 'b': 2}
dict6.update(c=3, d=4)
print("使用关键字参数更新:", dict6)  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# 使用可迭代对象
dict7 = {'x': 1}
dict7.update([('y', 2), ('z', 3)])
print("使用可迭代对象更新:", dict7)  # {'x': 1, 'y': 2, 'z': 3}


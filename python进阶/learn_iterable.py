# class my_iterable:
#     def __init__(self,start,end):
#         self.start=start       # start，end代表数据的起始位置，可以是更广泛的数据形式如列表
#         self.end=end
#
#     def __iter__(self):
#         return my_iterator(self.start,self.end)
#
# class my_iterator:
#     def __init__(self,current,end):
#         self.current=current
#         self.end=end
#     def __next__(self):
#         if self.current<self.end:  # 首先要判断还有数据
#             value=self.current     # 将当前的数据赋值给变量返回
#             self.current+=1        # 数据位置加1
#             return value
#         else:
#             raise  StopIteration
#
#
# iterable1=my_iterable(1,3)
# iterator1=iter(iterable1)
# print('第一次实例化迭代器',next(iterator1))
# print(next(iterator1)) # 自动计数累加
# print(next(iterator1)) # 自动计数累加
# iterator2=iter(iterable1)  # 计数器会初始化
# print('第二次实例化迭代器',next(iterator2))
#
# for iter_num in iterable1:
#     print('iter_num',iter_num)
#
#
# # 迭代器也是可迭代对象
#
# class MyRange:
#     def __init__(self, start, end):
#         self.start = start
#         self.end = end
#         self.current = start
#     def __iter__(self):
#         return self # 返回自身，因为自身就是迭代器
#     def __next__(self):
#         if self.current < self.end:
#             value = self.current
#             self.current += 1
#             return value
#         else:
#             raise StopIteration # 使用
# my_range = MyRange(1,4)
# for num in my_range:
#     print(num) # 输出1,2,3
# # 再次遍历就不会有输出了，因为迭代器已经到头
# print("再次遍历:")
# for num in my_range:
#     print(num) # 不会输出```
# #
#
#
# # 生成器函数和生成器对象
# def simple_generator():
#     yield 1
#     yield 2
#     yield 3
# # 创建生成器对象
# gen=simple_generator()
# #开始迭代
# print(next(gen))
# print(next(gen))
# print(next(gen))
# try:
#     print(next(gen))
# except StopIteration:
#     print("生成器已经耗尽")
#
# for x in simple_generator():
#        print(x)
# #
# # enumerate函数的用法
#
numbers = [1, 2, 3, 4, 5]
for i ,j in enumerate(numbers):
    print(i,j)

print(list(enumerate(numbers,start=2)))
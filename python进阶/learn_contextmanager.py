# class FileManager:
#     def __init__(self,file_name,mode):
#         self.file_name=file_name
#         self.mode=mode
#         self.file=None
#
#     def __enter__(self):
#         print("打开文件...")
#         self.file=open(self.file_name,self.mode)
#         return self.file  # 返回资源
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         print("关闭文件...")
#         if self.file:
#             self.file.close()
#             # 处理异常（返回True则阻止异常传播）
#         if exc_type:
#             print(f"发生异常: {exc_type}")
#         return False  # 不屏蔽异常
#
#
# # 使用示例
# with FileManager("test.txt", "w") as f:
#     f.write("Hello Context!")  # 即使发生异常，文件也会关闭
#     # 触发异常测试: 1/0
# print("操作完成")
#
#
# # 传统方式（易出错）
# file = open('test.txt')
# try:
#     data = file.read() # 若此处抛出异常，
#     file.close()      # 可能不被执行！
# finally:
#     file.close()      # 必须手动确保关闭
#
#
#
from contextlib import contextmanager
@contextmanager
def file_manager(filename, mode):
    try:
        print("打开文件...")
        f = open(filename, mode)
        yield f  # 返回资源
    except Exception as e:
        print(f"发生异常: {type(e).__name__}")
    finally:
        print("关闭文件...")
        f.close()


# 使用示例
with file_manager("test.txt", "a") as f:
    f.write("\nAppend content")
    # 触发异常测试: int('abc')
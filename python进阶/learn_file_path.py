import os
from pathlib import Path
import shutil

# 拼接路径
file_path1 = os.path.join("data", "documents1", "test1.txt")
print('file_path1:',file_path1)

file_path2 = Path("data2")/ "documents2" / "test2.txt" # 自动处理路径分隔符
print('file_path2:',file_path2)

dir_path3 = os.path.join('data', 'documents1')
print('file_path3:',dir_path3)

dir_path4 = Path("data2")/ "documents2"
print('file_path4:',dir_path4)
#
#
# # 创建目录
os.makedirs(dir_path3, exist_ok=True)
dir_path4.mkdir(parents=True, exist_ok=True)

#
# #
# # 创建文件
# # （1）创建空文件
# 方法1：使用open()
open(file_path1, 'w').close()
# 方法2：使用pathlib
file_path2.touch()  # touch()方法模拟Unix的touch命令
# # # （2）写入文件
# # # 使用内置的 open() 函数（最常用）
# # # 通过写入模式打开文件，如果文件不存在会自动创建。
# 方法1：文本模式创建（默认）
with open(file_path1, 'w', encoding='utf-8') as f:
           f.write('这是通过open()创建的文本文件')
#
# # write_text（）
# 方法2：直接写入内容
file_path2.write_text('这是通过pathlib创建的文本文件', encoding='utf-8')
#
# # # 路径是否存在
# 包含文件
exist1=os.path.exists(file_path1)
print('exist1:',exist1)
exist2=file_path2.exists()
print('exist2:',exist2)
#
# # 纯目录
exist3=os.path.exists(dir_path3)
print('exist3:',exist3)
exist4=dir_path4.exists()
print('exist4:',exist4)
#
# #
# # 获取文件名（带扩展）
file_name_ext1=os.path.basename(file_path1)
print('file_name_ext1:',file_name_ext1)
file_name_ext2=file_path2.name
print('file_name_ext2:',file_name_ext2)

# 获取文件名（不带扩展）
file_name_no_ext1=os.path.basename(file_path1).split('.')[0]
print('file_name_no_ext1:',file_name_no_ext1)
file_name_no_ext2=file_path2.stem
print('file_name_no_ext2:',file_name_no_ext2)
#
# # # 获取目录
file_dir1=os.path.dirname(file_path1)
print('file_dir1:',file_dir1)

file_dir2=file_path2.parent
print('file_dir2:',file_dir2)

file_dir3=os.path.dirname(dir_path3)
print('file_dir3:',file_dir3)
#
# # 获取当前目录
current_dir1 = os.getcwd()
print('current_dir1:', current_dir1)
# 获取绝对路径
print('__file__值:',__file__)



# 获取绝对路径
relative_path = 'relative_path'
abs_path = os.path.abspath(relative_path)
print('abs_path', abs_path)
# abs_path2=Path(__file__).resolve()
abs_path2 = Path('data2/test2.docx').resolve()
print('abs_path2', abs_path2)
#
#
print('是否绝对路径：',os.path.isabs(file_path1))
if os.path.isabs(abs_path2):

    rel_path = Path(abs_path2).relative_to(current_dir1)
    print('rel_path',rel_path)
#

# 目录的遍历 os.listdir和os.walk
list_dir=os.listdir('data')
print('list_dir:',list_dir)

for tuple_path in os.walk('data'):
    print(tuple_path)
# #
#
# 1. 删除文件：不能删除目录
os.remove(file_path1)
file_path2.unlink()
# # 2.删除空目录：
os.rmdir(dir_path3)
dir_path4.rmdir()
# # 3. 删除目录（包括非空目录）：
shutil.rmtree('data')
shutil.rmtree('data2')

# Windows 路径示例
win_path = Path(r"C:\Users\Alice\文档\file.txt")
print(win_path.as_posix())
# 输出: C:/Users/Alice/文档/file.txt
# Linux/macOS 路径示例
linux_path = Path("/home/alice/文档/file.txt")
print(linux_path.as_posix())
# 输出: /home/alice/文档/file.txt
# 混合路径示例
mixed_path = Path("C:\\Users/Bob\\文件//report.docx")
print(mixed_path.as_posix())
# 输出: C:/Users/Bob/文件/report.docx
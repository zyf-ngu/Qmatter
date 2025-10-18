# import argparse
#
# # 1创建参数解析器：是"配置器"，负责定义参数规则，在下面的添加参数后基本不变，ArgumentParser 类实例
# parser = argparse.ArgumentParser()
#
# # 解析器添加参数：位置参数和可选选项参数
#
# # parser.add_argument('file_name', action='store', help='要处理的文件名')
# # parser.add_argument(
# #     '-v', '--versbose', action='store_true')
#
# parser.add_argument('file_name',nargs='?', action='store', type=int, help='要处理的文件名')
# parser.add_argument('-v','--versbose' ,action='store',dest='new_name')
#
# # 输出解析结果
# args = parser.parse_args()
# print(type(args))
# print(args.file_name)
# print(type(args.file_name))
#
# # print(args.versbose)
#
# print(args.new_name)


from config import DATABASE, DEBUG

print(DATABASE["host"])  # 输出: localhost
print(DEBUG)  # 输出: True

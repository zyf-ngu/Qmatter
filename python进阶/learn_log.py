import logging

# 1. 创建Logger实例（命名为 'my_app'）
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)  # 设置最低处理级别（DEBUG及以上都会处理）

# 2. 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # 控制台只输出WARNING及以上级别

# 3. 创建文件处理器
file_handler = logging.FileHandler('app.log', mode='w')  # 写入模式
file_handler.setLevel(logging.DEBUG)  # 文件记录所有DEBUG及以上级别

# 4. 创建格式化器
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 5. 将格式化器添加到处理器
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 6. 将处理器添加到Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ============ 测试日志输出 ============
logger.debug('这是一条调试信息')      # 仅写入文件
logger.info('程序正常启动')           # 仅写入文件
logger.warning('磁盘空间不足80%!')    # 同时输出到控制台和文件
logger.error('无法连接数据库!')       # 同时输出到控制台和文件
logger.critical('系统崩溃!!!')       # 同时输出到控制台和文件
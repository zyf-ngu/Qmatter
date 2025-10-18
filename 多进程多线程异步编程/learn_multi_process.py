# # 基本用法
# import multiprocessing
# from multiprocessing import Process,Manager
# from typing import List
# import time
#
#
# def calculate_squares(nums:List,shared_queue):
#     for num in nums:
#         # print('num:',num)
#         shared_queue.put(num**2)
#
#
# if __name__=='__main__':
#     numbers=[1,2,3,4]
#     shared_queue = multiprocessing.Queue()  # 进程安全队列
#
#     p1=Process(target=calculate_squares,args=(numbers[:2],shared_queue))
#     p2=Process(target=calculate_squares,args=(numbers[2:],shared_queue))
#
#     p1.start() # 启动进程1
#     p2.start()  # 启动进程2
#
#     p1.join()  # 等待进程1结束
#     p2.join()  # 等待进程2结束
#     while not shared_queue.empty():
#         print(shared_queue.get())


#
# # 启动方法
# import multiprocessing
# import os
#
#
# def worker():
#     print(f"子进程ID: {os.getpid()}, 父进程ID: {os.getppid()}")
#
#
# if __name__ == "__main__":
#     # 设置进程创建方式（必须在创建进程前调用）
#     multiprocessing.set_start_method('spawn')  # 可选 'fork'/'spawn'/'forkserver'
#
#     # 创建并启动子进程
#     p = multiprocessing.Process(target=worker)
#     p.start()
#     p.join()
#     print(f"主进程ID: {os.getpid()}")




# # 启动事件
# import multiprocessing
# from multiprocessing import Process
# import time
# def event_func(start_event,id):
#     print(f"Worker-{id} waiting to start...")
#     start_event.wait()
#     print(f"Worker-{id} started at {time.time():.2f}")
#
# if __name__=='__main__':
#     start_event=multiprocessing.Event()
#     processes=[]
#     for i in range(3):
#         p=Process(target=event_func,args=(start_event,i))
#         processes.append(p)
#     for p in processes:
#         p.start()
#
#     print("Main process initializing resources...")
#     time.sleep(2)
#     # 设置事件唤醒所有工作进程 (关键位置!)
#     print("Main process setting event at", time.time())
#     start_event.set()  # 所有等待的worker同时唤醒
#
#     # 等待子进程结束
#     # for p in processes:
#     #     p.join()
#     print("All workers completed")



# # 进程池
# import multiprocessing
# import time
# def square(x):   #后面演示代码均使用此执行函数
#    time.sleep(0.2 - x*0.01) # 数字越大执行越快
#    print(f"处理: {x}") # 显示处理顺序
#    return x * x
#
#
# if __name__ == '__main__':
    # with multiprocessing.Pool(4) as pool: # map 阻塞直到所有任务完成
    #     results = pool.map(square, range(10))
    #     print("主进程不可以继续执行其他任务...")
    #     print("map results:", results)
    #
    #     async_result = pool.map_async(square, range(10))  # 非阻塞异步映射
    #     print("主进程可以继续执行其他任务...")
    #     time.sleep(0.2)  # 模拟其他工作
    #     # 需要结果时调用 get()，会阻塞直到结果就绪
    #     results = async_result.get()
    #     print("map_async results:", results)  # 输出: 主进程可以继续执行其他任务...

#     with multiprocessing.Pool(4) as pool:  # 返回按输入顺序排序的迭代器
#         results_iter = pool.imap(square, range(10))
#         print("开始获取结果:")
#         for i, result in enumerate(results_iter, 1):
#             print(f"获取结果: {result}")
#
#     with multiprocessing.Pool(4) as pool:  # 按完成顺序返回结果
#         results_iter = pool.imap_unordered(square, range(10))
#     print("开始获取结果 (按完成顺序):")
#     for i, result in enumerate(results_iter, 1):
#         value, squared = result
#     print(f"获取结果 : {value}² = {squared}")
#
#
#
# def power(base, exponent):
#     return base ** exponent
# if __name__ == '__main__':
#     with multiprocessing.Pool(4) as pool:
#     # 参数是元组列表，每个元组包含多个参数
#         arguments = [(2, 3), (3, 2), (4, 2), (5, 3)]
#         results = pool.starmap(power, arguments)
#         print("starmap results:", results)
#
#     with multiprocessing.Pool(4) as pool:
#         arguments = [(2, 3), (3, 2), (4, 2), (5, 3)]
#         async_result = pool.starmap_async(power, arguments)
#         print("主进程执行其他任务...")
#         # 等待结果就绪
#         results = async_result.get()
#         print("starmap_async results:", results)
#         # 输出:    主进程执行其他任务...   results: [8, 9, 16, 125]

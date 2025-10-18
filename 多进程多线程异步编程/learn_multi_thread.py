import threading
# # 共享资源
# counter = 0
# # 创建锁对象
# lock = threading.Lock()
# def increment_without_lock():
#      """不加锁的计数器增加"""
#      global counter#全局共享资源
#      for _ in range(1000):
#         temp = counter
#         temp += 1
#         counter = temp
# def increment_with_lock():
#     """加锁的计数器增加"""
#     global counter#全局共享资源
#     for _ in range(1000):
# #在对共享资源操作前获取锁，最后释放锁
#         lock.acquire()
#         try:
#             temp = counter
#             temp += 1
#             counter = temp
#         finally:
#             lock.release()
# def run_demo(func):
#     """测试函数"""
#     global counter
#     counter = 0 # 重置计数器
#     # 创建5个线程
#     threads = [threading.Thread(target=func) for _ in range(5)]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#     print(f"最终计数器值: {counter}")
#
# print("不加锁情况：")
# run_demo(increment_without_lock) # 输出通常小于5000
# print("\n加锁情况：")
# run_demo(increment_with_lock) # 正确输出5000


# 线程锁
# 创建两个不同的锁对象
# lock1 = threading.Lock()
# lock2 = threading.Lock()
# print(id(lock1)) # 输出锁1的内存地址（例如：140735812859664）
# print(id(lock2)) # 输出锁2的内存地址（不同于锁1）
#
# def task1():
#     with lock1:
#         # 使用锁1
#         print("Task1 获得了 lock1")
# def task2():
#     with lock1:
#          # 使用同一个锁1（同一个对象）
#         print("Task2 获得了 lock1")
# def task3():
#     with lock2:
#         # 使用不同的锁2
#         print("Task3 获得了 lock2")
# # 创建线程
# t1 = threading.Thread(target=task1)
# t2 = threading.Thread(target=task2)
# t3 = threading.Thread(target=task3)
# t1.start()
# t2.start()
# t3.start()
#
#
# import threading
# import time
# class ThreadSafeCounter:
#     def __init__(self):
#         self.value = 0  # 计数器值
#         # 创建RLock原子锁（允许同一线程重入）
#         self.atomic = threading.RLock()
#     def increment(self):
#         # 获取RLock锁（同一线程可多次调用）
#         with self.atomic:  # 第一次获取锁
#             temp = self.value  # 读取临时值
#             time.sleep(0.001)  # 模拟处理延迟
#             # 嵌套获取同一锁（RLock允许）
#             with self.atomic:  # 第二次获取锁（计数器+1）
#                 self.value = temp + 1  # 安全更新值
# # 创建计数器实例
# counter = ThreadSafeCounter()
# # 定义线程任务函数
# def worker(counter_obj: ThreadSafeCounter, n: int):
#     for _ in range(n):
#         counter_obj.increment()
# # 创建两个线程
# threads = [
#     threading.Thread(target=worker, args=(counter, 100)),  threading.Thread(target=worker, args=(counter, 100))  ]
# # 启动线程
# for t in threads:
#     t.start()
# # 等待线程结束
# for t in threads:
#     t.join()
# # 输出结果
# print(f"Final counter value: {counter.value}")
# # 预期输出：200
#
#
# # 线程池
# from concurrent.futures import ThreadPoolExecutor
# import time
# def square(x):
#     time.sleep(0.1)
# # 模拟I/O操作
#     return x * x
# if __name__ == '__main__':
#     with ThreadPoolExecutor(max_workers=4) as executor:
#     # map 阻塞直到所有任务完成
#         results = list(executor.map(square, range(10)))
#         print("map results:", results) # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] ```
#
#
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import random
# def square(x):
#     sleep_time = random.uniform(0.05, 0.2)
#     time.sleep(sleep_time)
#     return (x, x * x, sleep_time)
# if __name__ == '__main__':
#     with ThreadPoolExecutor(max_workers=4) as executor:
#     # 提交任务
#         futures = [executor.submit(square, i) for i in range(10)]
#         print("按完成顺序获取结果:")
#     #    as_completed 返回一个迭代器，在任务完成时产生future对象
#         for future in as_completed(futures):
#             x, result, sleep_time = future.result()
#             print(f"任务 {x} = {result} (耗时 {sleep_time:.3f}s)")
#
#
# from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
# import time
# def square(x):
#     time.sleep(0.1 * (10 - x))# 数字越大执行越快
#     return x * x
# if __name__ == '__main__':
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         # 提交10个任务
#         futures = [executor.submit(square, i) for i in range(10)]
#          # 等待至少一个任务完成
#         done, not_done = wait(futures, return_when=FIRST_COMPLETED)
#         print(f"\n一个任务完成: 已完成 {len(done)} 个, 未完成 {len(not_done)} 个")
#         for f in done:
#             print(f"结果: {f.result()}")
#             # 再等待全部完成
#             done, not_done = wait(futures)
#         print(f"\n所有任务完成: 已完成 {len(done)} 个")
#         results = [f.result() for f in futures]
#         print("所有结果:", results)
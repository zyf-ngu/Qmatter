# import asyncio
# import time
#
# async def func1():
#     print('协程1开始工作')
#     await asyncio.sleep(1)  # 异步函数里需要有“等待”的语句,await等待协程对象执行完毕
#     print('协程1继续工作')
#
# async def func2():
#     print('协程2开始工作')
#     await asyncio.sleep(1)  # 异步函数里需要有“等待”的语句,await等待协程对象执行完毕
#     print('协程2继续工作')
#     return 'func2'
#
# async def main():
#     task1=asyncio.create_task(func1())
#     task2=asyncio.create_task(func2())
#     await task1 # 等待任务对象才是并发执行，如果任务对象有返回结果,需要接收
#     result= await task2
#     print(result)
#
# if __name__=='__main__':
#     asyncio.run(main())



#
import asyncio
import time


async def func1():
    print('协程1开始工作')
    await asyncio.sleep(1)  # 异步函数里需要有“等待”的语句,await等待协程对象执行完毕
    print('协程1继续工作')


async def func2():
    print('协程2开始工作')
    await asyncio.sleep(1)  # 异步函数里需要有“等待”的语句,await等待协程对象执行完毕
    print('协程2继续工作')
    return 'func2'


async def main():
    # task1 = asyncio.create_task(func1())
    # task2 = asyncio.create_task(func2())
    # await task1  # 等待任务对象才是并发执行，如果任务对象有返回结果,需要接收
    # result = await task2
    # print(result)

    tasks3 = [
        asyncio.create_task(func1()),
        asyncio.create_task(func2())
    ]
    await asyncio.wait(tasks3)  # asyncio.wait等待任务列表


async def main2():
    # await tasks1
    await func1()  # 直接等待协程对象是顺序执行，不是并发执行
    await func2()


if __name__ == '__main__':
    asyncio.run(main())
    # asyncio.run(main2())

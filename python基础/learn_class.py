# #
# 类属性和实例属性
# class BaseClass:
#     # 类属性
#     class_attribute = '这是类属性'
#
#     def __init__(self, instance_attribute):
#         # 实例属性
#         self.instance_attribute1 = instance_attribute  # 通过形参赋值，类实例化时传入
#         self.__instance_attribute2 = 0  # 可以直接初始化，不用通过形参赋值,且是私有属性，实例名和类名不能直接访问
#
#     def use_class_attribute(self):
#         print(self.class_attribute)
#
#
# # 类实例化，类名后的参数需要和__init__方法的形参一致
# obj1 = BaseClass(instance_attribute='instance_value1')
# obj2 = BaseClass(instance_attribute='instance_value2')
# obj2.use_class_attribute()
# # 类属性访问,包括类名和实例名
# print('类名访问类属性：', BaseClass.class_attribute)
# print('实例名访问类属性：', obj1.class_attribute)
# print('实例名访问类属性：', obj2.class_attribute)
#
# # 实例属性访问
# # print('类名访问实例属性：',BaseClass.instance_attribute1) # 报错，类名不能访问实例属性
# print('实例名访问实例属性：', obj1.instance_attribute1)
# print('实例名访问实例属性：', obj2.instance_attribute1)
# # print('实例名访问实例属性：',obj2.instance_attribute2) # 报错，私有属性不能直接访问
# #
#
# # 类属性修改
# BaseClass.class_attribute = '类直接修改后的类属性'
# print('实例名访问修改后的类属性：', obj1.class_attribute)  # 输出：类直接修改后的类属性
# obj1.class_attribute = '实例直接修改后的类属性'
# print('另一个实例名访问实例修改后的类属性：', obj2.class_attribute)  # 输出：类直接修改后的类属性
#
# # 实例属性修改
# obj1.instance_attribute1 = '实例修改后的实例属性'
# print('实例名访问修改后的实例属性：', obj1.instance_attribute1)  # 输出：实例修改后的实例属性
# print('另一个实例名访问修改后的实例属性：', obj2.instance_attribute1)  # 输出：instance_value2（实例属性没变）
#
#
# #
#
#
# 类的方法
# class Student:
#     # 类属性
#     school_name = "Python University"
#
#     def __init__(self, name, age):
#         """初始化方法/构造方法"""
#         # 实例属性
#         self.name = name
#         self.age = age
#         print(f"创建了学生: {self.name}")
#
#     def display_info(self):
#         """普通方法/实例方法 - 需要访问实例属性"""
#         print(f"学生姓名: {self.name}, 年龄: {self.age}")
#
#     @classmethod
#     def change_school(cls, new_school):
#         """类方法 - 操作类属性"""
#         cls.school_name = new_school
#         print(f"学校已更改为: {cls.school_name}")
#
#     @staticmethod
#     def is_adult(age):
#         """静态方法 - 不依赖实例或类状态"""
#         return age >= 18
#
#
# # 创建实例，自动调用 __init__ 方法
# student1 = Student("小明", 20)
#
# # 调用普通方法
# student1.display_info()
#
# # 调用静态方法
# print(f"20岁是成年人吗? {Student.is_adult(20)}")
#
# # 调用类方法修改类属性
# Student.change_school("C++")
# print(f"学校名称: {Student.school_name}")

#
#
#
#
#
#
#
# # 继承的种类
# 1单继承：只有一个父类
# class Animal:
#     def speak(self):
#         print('动物会发出声音')
# class Dog(Animal):
#     def spark(self):
#         print('小狗会汪汪叫')
# dog = Dog()
# dog.speak()  # 调用父类的方法， 输出：动物会发出声音
# dog.spark()  # 调用自己的方法，输出：小狗会汪汪叫
#
#
#
# class Parent:
#     def methodA(self):
#         print("Parent: methodA called")
#         self.methodB()  # 此处实际调用子类重写的 methodB
#
#     def methodB(self):
#         print("Parent: methodB called")
#
#
# class Child(Parent):
#     def methodB(self):  # 重写父类的 methodB
#         print("Child: methodB called")
#
#
# # 测试
# child = Child()
# child.methodA()  # 调用父类的 methodA 但输出 Child: methodB called  # 实际调用子类重写的 methodB
#
# # 多继承：子类同时继承多个父类（Python 特色）
# class Flyer:
#     def fly(self):
#         print("在空中飞行")
# class Swimmer:
#     def swim(self):
#         print("在水里游泳")
# class Duck(Flyer, Swimmer):  # 多继承
#     def speak(self):
#         print("嘎嘎嘎!")
#
# duck = Duck()
# duck.fly()  # 来自 Flyer -> 在空中飞行
# duck.swim()  # 来自 Swimmer -> 在水里游泳
# duck.speak()  # 子类自有方法 -> 嘎嘎嘎!
#
#
# # 多层继承;形成继承链（祖父 → 父 → 子）
# class Vehicle:
#     def transport(self):
#         print("运输工具")
# class Car(Vehicle):
#     def run(self):
#         print("在公路上行驶")
# class ElectricCar(Car):  # 多层继承
#     def charge(self):
#         print("电能驱动")
# tesla = ElectricCar()
# tesla.transport()  # 继承自 Vehicle -> 运输工具
# tesla.run()  # 继承自 Car -> 在公路上行驶
# tesla.charge()  # 自有方法 -> 电能驱动
#
#
# # 菱形继承：父类有共同祖先
# class A:
#     def show(self):
#         print('A')
# class B(A):
#     def show(self):
#         print('B')
# class C(A):
#     def show(self):
#         print('C')
# class D(B, C):
#     # def show(self):
#     #     print('D')  # （1）如果子类重写方法，则直接调用自己的方法
#     pass  # （2）如果子类不重写方法，则按照MRO顺序调用父类祖先类方法
#
#
# d = D()
# d.show()  # 重写输出D；不重写：输出B
# print('D.mro:',
#       D.mro())  # D.mro: [<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>]
#
#
#
from abc import ABC, abstractmethod
class Shape(ABC):  # 抽象基类
    @abstractmethod
    def area(self):  # 抽象方法
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):  # 必须实现抽象方法
        return 3.14 * self.radius ** 2


# shape = Shape()  # 报错：不能实例化抽象类
circle = Circle(5)
print(circle.area())  # 78.5

#
#
# # super函数的对比
#
#
# super（）函数菱形继承问题解决方案

# （1）全部使用super()函数：则输出D-B-C-A
print('全部使用super()函数：则输出D-B-C-A')
class A:
    def process(self):
        print("Processing in A")


class B(A):
    def process(self):
        print("Processing in B")
        super().process()


class C(A):
    def process(self):
        print("Processing in C")
        super().process()


class D(B, C):
    def process(self):
        print("Processing in D")
        super().process()

d = D()
d.process()

# （2）所有的类都不使用super（）函数，父类名显式调用，D-B-A-C-A
print('所有的类都不使用super（）函数，父类名显式调用，D-B-A-C-A')
class A:
    def process(self):
        print("Processing in A")
class B(A):
    def process(self):
        print("Processing in B")
        A.process(self)
class C(A):
    def process(self):
        print("Processing in C")
        A.process(self)

class D(B, C):
    def process(self):
        print("Processing in D")
        B.process(self)
        C.process(self)


d = D()
d.process()


# （3）最后的子类D使用super（）函数，其它使用父类名称，：D-B-A
print('最后的子类D使用super（）函数，其它使用父类名称，：D-B-A')
class A:
    def process(self):
        print("Processing in A")


class B(A):
    def process(self):
        print("Processing in B")
        A.process(self)


class C(A):
    def process(self):
        print("Processing in C")
        A.process(self)


class D(B, C):
    def process(self):
        print("Processing in D")
        super().process()


d = D()
d.process()


# （4）在不同名称的方法内部调用父类的方法，D-B-C-A
print('（4）在不同名称的方法内部调用父类的方法，D-B-C-A')
class A:
    def process(self):
        print("Processing in A")
class B(A):
    def process(self):
        print("Processing in B")
        super().process()


class C(A):
    def process(self):
        print("Processing in C")
        super().process()


class D(B, C):
    def process2(self):
        print("Processing2 in D")
        super().process()


d = D()
d.process()


# （5）B或C不使用super（）函数，也不使用父类名称调用，D-B，停止到不继续调用的类，
print('（5）B或C不使用super（）函数，也不使用父类名称调用，D-B，停止到不继续调用的类，')
class A:
    def process(self):
        print("Processing in A")


class B(A):
    def process(self):
        print("Processing in B")


class C(A):
    def process(self):
        print("Processing in C")


class D(B, C):
    def process(self):
        print("Processing in D")
        super().process()


d = D()
d.process()

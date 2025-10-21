# 从自己创建的models库里导入ResNet模块
# import ResNet 仅仅是把ResNet.py导入进来,当我们创建ResNet的实例的时候需要通过指定ResNet.py中的具体类.
# 例如:我的ResNet.py中的类名是ResNet,则后面的模型实例化ResNet需要通过**ResNet.ResNet()**来操作
# 还可以通过 from 还可以通过 from ResNet import * 直接把ResNet.py中除了以 _ 开头的内容都导入
from tradtional_models.models.cv.ResNet import *
# torchvision：PyTorch的一个图形库，服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型
# transforms：主要是用于常见的一些图形变换
# datasets：包含加载数据的函数及常用的数据集接口
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# os：operating system（操作系统），os模块封装了常见的文件和目录操作
import os
import matplotlib.pyplot as plt

# # 设置数据转化方式，如数据转化为Tensor格式，数据切割等
# # Compose()：将多个transforms的操作整合在一起
# # ToTensor(): 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且归一化到[0,1.0]之间
# # compose的参数为列表[]
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),   # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为给定大小
#     transforms.RandomHorizontalFlip(p=0.5),   # 以0.5的概率竖直翻转给定的PIL图像
#     transforms.ToTensor(),               # 数据转化为Tensor格式
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 将图像三个通道的像素值归一化到[-1,1]之间，使模型更容易收敛
# ])
# test_transform = transforms.Compose([transforms.Resize((224, 224)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练集transform（适配单通道灰度图）
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 强制转为单通道
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # 缩放至64x64，避免过度拉伸
    transforms.RandomHorizontalFlip(p=0.5),  # 显式设置翻转概率
    transforms.RandomRotation(10),  # 新增随机旋转，增强泛化性
    transforms.ToTensor(),
    # 基于FER2013统计的均值/方差（更精准的归一化，而非默认0.5）
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道仅需1个均值/方差
])

# 测试集transform（无数据增强，仅固定尺寸）
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),  # 固定尺寸，与训练一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ImageFolder(root, transform``=``None``, target_transform``=``None``, loader``=``default_loader)
# root 指定路径加载图片;  transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
# target_transform：对label的转换   loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象
# label是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}，一般来说最好直接将文件夹命名为从0开始的数字，举例来说，两个类别，
# 狗和猫，把狗的图片放到文件夹名为0下；猫的图片放到文件夹名为1的下面。
# 这样会和ImageFolder实际的label一致， 如果不是这种命名规范，建议看看self.class_to_idx属性以了解label和文件夹名的映射关系
# python中\是转义字符，Windows 路径如果只有一个\，会把它识别为转义字符。
# 可以用r''把它转为原始字符，也可以用\\,也可以用Linux的路径字符/。
train_dataset = ImageFolder(r"E:\计算机\data\fer2013_数据增强版本\train", train_transform)
test_dataset = ImageFolder(r"E:\计算机\data\fer2013_数据增强版本\test", test_transform)

# DataLoader：将读取的数据按照batch size大小封装并行训练
# dataset (Dataset)：加载的数据集
# batch_size (int, optional)：每个batch加载多少个样本(默认: 1)
# shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据(默认: False)

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 优化后
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=os.cpu_count()-1,  # 多进程加速
    pin_memory=True  # 锁页内存，加速GPU传输
)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

blocks_num = [3, 4, 6, 3]
model = ResNet(BottleBlock, blocks_num, num_classes=7).to(device)
# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()
# 定义adam优化器
# params(iterable)：要训练的参数，一般传入的是model.parameters()
# lr(float)：learning_rate学习率，也就是步长，默认：1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 优化后新增
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)  # 当准确率不再提升时自动降学习率



# 迭代次数（训练次数）
epochs = 30
# 用于判断最佳模型
best_acc = 0.0
# 最佳模型保存地址
# save_path = './{}Net.pth'.format(model_name)
train_steps = len(train_dataloader)


def train(train_dataloader, model, loss_fn, optimizer):
    loss, acc, n = 0.0, 0.0, 0
    # dataloader: 传入数据（数据包括：训练数据和标签）
    # enumerate()：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，一般用在for循环当中
    # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
    # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型的）
    # 和dataloader结合使用时返回数据下标从0开始，一般使用batch变量承接，最大数为样本总数除以batch size大小，
    # （在创建dataloader时会把batch size作为参数传入），下标对应的数据是一batch的数据和标签，以元组的形式返回
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        cur_loss = loss_fn(output, y)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率
        # output.shape[0]一维长度为该批次的数量
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc = torch.sum(pred == y) / output.shape[0]
        # 清除过往梯度值，防止上个batch的数据的梯度值累积
        optimizer.zero_grad()
        cur_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # .item()将张量转化为标量
        loss += cur_loss.item()
        acc += cur_acc.item()
        n = n + 1
    train_loss = loss / n
    train_acc = acc / n
    print('train_loss==' + str(train_loss))
    # 计算训练的准确率
    print('train_acc' + str(train_acc))
    return train_loss, train_acc


# 测试函数里参数无优化器，不需要再训练，只需要测试和验证即可
def test(test_dataloader, model, loss_fn):
    loss, acc, n = 0.0, 0.0, 0
    model.eval()  # 测试阶段必须设置为评估模式（关闭Dropout/BN更新）
    # 将with语句包裹起来的部分停止梯度的更新，从而节省了GPU算力和显存，但是并不会影响dropout和BN层的行为
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(pred == y) / output.shape[0]
            loss += cur_loss.item()
            acc += cur_acc.item()
            n = n + 1
        model.train()  # 恢复训练模式，避免影响后续训练
        test_loss = loss / n
        test_acc = acc / n
        print('test_loss==' + str(test_loss))
        # 计算训练的准确率
        print('test_acc' + str(test_acc))
        return test_loss, test_acc


def matplot_loss(train_loss, test_loss):
    fig, ax = plt.subplots(1, 1)
    # 参数label = ''传入字符串类型的值，也就是图例的名称
    ax.plot(train_loss, label='train_loss')
    ax.plot(test_loss, label='test_loss')
    # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）
    ax.legend(loc='best')
    ax.set_xlabel('epoch')  # x轴为轮次
    ax.set_ylabel('loss')  # y轴为损失值

    ax.set_title("训练集和验证集的loss值对比图")
    plt.show()

    # 准确率


def matplot_acc(train_acc, test_acc):
    fig, ax = plt.subplots(1, 1)
    ax.plot(train_acc, label='train_acc')
    ax.plot(test_acc, label='test_acc')
    ax.legend(loc='best')
    ax.set_xlabel('acc')
    ax.set_ylabel('epoch')
    ax.set_title("训练集和验证集的acc值对比图")
    plt.show()


epochs = 20

loss_train = []
acc_train = []
loss_test = []
acc_test = []
best_acc = 0.0  # 初始化为0，而非min_acc（语义更清晰）
save_dir = "save_model"
os.makedirs(save_dir, exist_ok=True)  # 简化创建目录的逻辑

for t in range(epochs):
    # 不同的优化函数不同的使用方法
    # lr_scheduler.step()
    print(f"{t + 1}\n------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test(test_dataloader, model, loss_fn)
    lr_scheduler.step(test_acc)  # 基于测试准确率调整学习率
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_test.append(test_loss)
    acc_test.append(test_acc)
    # 仅保存测试准确率最高的模型
    if test_acc > best_acc:
        best_acc = test_acc
        # 保存字典包含多类信息，便于断点续训
        save_dict = {
            "epoch": t + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "train_loss": train_loss,
            "test_loss": test_loss
        }
        # 删除旧的最优模型（避免冗余）
        old_models = [f for f in os.listdir(save_dir) if f.startswith("resnet-best")]
        for f in old_models:
            os.remove(os.path.join(save_dir, f))
        # 保存新的最优模型
        torch.save(save_dict, os.path.join(save_dir, f"resnet-best-epoch{t + 1}-acc{best_acc:.4f}.pth"))
        print(f"Save best model (epoch {t + 1}, acc {best_acc:.4f})")

matplot_loss(loss_train, loss_test)
matplot_acc(acc_train, acc_test)

# def main():
#     # 如果有NVIDA显卡，转到GPU训练，否则用CPU
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("using {} device.".format(device))
#
# }
#
#     # abspath()：获取文件当前目录的绝对路径
#     # join()：用于拼接文件路径，可以传入多个路径
#     # getcwd()：该函数不需要传递参数，获得当前所运行脚本的路径
#     #data_root = os.path.abspath(os.getcwd())
#     # 得到数据集的路径
#     #image_path = os.path.join(data_root, "flower_data")
#     # exists()：判断括号里的文件是否存在，可以是文件路径
#     # 如果image_path不存在，则会抛出AssertionError错误，报错为参数内容“ ”
#    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
# # 训练集长度
# train_num = len(train_dataset)
#
# # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
# # class_to_idx：获取分类名称对应索引
# flower_list = train_dataset.class_to_idx
# # dict()：创建一个新的字典
# # 循环遍历数组索引并交换val和key的值重新赋值给数组，这样模型预测的直接就是value类别值
# cla_dict = dict((val, key) for key, val in flower_list.items())
# # 把字典编码成json格式
# json_str = json.dumps(cla_dict, indent=4)
# # 把字典类别索引写入json文件
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)
#
# # 一次训练载入32张图像
# batch_size = 32
# # 确定进程数
# # min()：返回给定参数的最小值，参数可以为序列
# # cpu_count()：返回一个整数值，表示系统中的CPU数量，如果不确定CPU的数量，则不返回任何内容
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
# print('Using {} dataloader workers every process'.format(nw))
# # DataLoader：将读取的数据按照batch size大小封装给训练集
# # dataset (Dataset)：输入的数据集
# # batch_size (int, optional)：每个batch加载多少个样本，默认: 1
# # shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据，默认: False
# # num_workers(int, optional): 决定了有几个进程来处理，默认为0意味着所有的数据都会被load进主进程
# train_bar = tqdm(train_loader, file=sys.stdout)
# train_bar: 传入数据（数据包括：训练数据和标签）
# enumerate()：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
# enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
# x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型）
# # 测试集长度
# val_num = len(validate_dataset)
#
# validate_loader = torch.utils.data.DataLoader(validate_dataset,
# batch_size = batch_size, shuffle = False,
# num_workers = nw)
# print("using {} images for training, {} images for validation.".format(train_num,
#                                                                        val_num))

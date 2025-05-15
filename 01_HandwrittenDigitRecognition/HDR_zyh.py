# -*- coding:utf-8 -*- #
#   name: HDR_zyh
#   time: 2025/4/11 16:46
# author: zhangyehao_202428010315039
#  email: 3074675457@qq.com
import torch
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils import data
from torchvision import transforms
import netron
import torch.onnx

def create_batch_data(batch_size_in, resize=None):
    """
    使用DataLoader创建随机批量数据
    :param batch_size_in: 批量大小
    :param resize: 可选参数，用于调整图像大小
    :return: 训练数据加载器和测试数据加载器
    """
    # 定义图像转换
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 加载MNIST数据集
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, transform=trans, download=True
    )

    # 创建数据加载器
    return (
        data.DataLoader(mnist_train, batch_size_in, shuffle=True, num_workers=4),
        data.DataLoader(mnist_test, batch_size_in, shuffle=False, num_workers=4)
    )


class ZyhNet(nn.Module):
    """
    构建卷积神经网络，与LeNet类似
    架构如下：
    - 卷积层1：输入通道数为1，输出通道数为6，卷积核大小为3x3，步幅为1，填充为1
    - ReLU激活函数
    - 平均池化层：池化窗口大小为2x2，步幅为2
    - 卷积层2：输入通道数为6，输出通道数为16，卷积核大小为5x5，步幅为1
    - ReLU激活函数
    - 平均池化层：池化窗口大小为2x2，步幅为2
    - 展平层
    - 全连接层1：输入特征数为400，输出特征数为120
    - ReLU激活函数
    - 全连接层2：输入特征数为120，输出特征数为84
    - ReLU激活函数
    - 全连接层3：输入特征数为84，输出特征数为10（对应10个数字分类）
    """

    def __init__(self):
        super(ZyhNet, self).__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),  # 输出6*28*28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 输出6*14*14
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # 输出16*10*10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 输出16*5*5
        )
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    @staticmethod
    def reshape_(z):
        """
        重塑输入张量的形状以匹配网络的输入要求
        :param z: 输入张量
        :return: 重塑后的张量
        """
        return z.reshape(-1, 1, 28, 28)

    def forward(self, m):
        """
        定义前向传播过程
        :param m: 输入张量
        :return: 网络输出
        """
        m = self.reshape_(m)
        m = self.CNN1(m)
        m = self.CNN2(m)
        m = self.FC1(m)
        return m


def same_number(y_hat_in, y):
    """
    返回预测值与真实值相等的个数
    :param y_hat_in: 预测值
    :param y: 真实值
    :return: 预测值与真实值相等的个数
    """
    y_hat_in = torch.tensor(y_hat_in) if not isinstance(y_hat_in, torch.Tensor) else y_hat_in
    y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
    if len(y_hat_in.shape) > 1 and y_hat_in.shape[1] > 1:
        y_hat_in = y_hat_in.argmax(dim=1)
    cmp = y_hat_in == y
    return float(cmp.sum().item())


class Accumulator:
    """
    构建n列变量，每列累加
    :param n: 变量列数
    """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        """
        累加数据
        :param args: 要累加的值
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """
        重置累加的数据
        """
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的数据
        :param index: 数据索引
        :return: 累加的数据
        """
        return self.data[index]


def test_net(net_in, testdata_in, device_in):
    """
    计算测试误差
    :param net_in: 输入网络
    :param testdata_in: 测试数据加载器
    :return: 测试准确率
    """
    if isinstance(net_in, torch.nn.Module):
        net_in.eval()
    metric = Accumulator(2)
    for x_in, y_in in testdata_in:
        x_in, y_in = x_in.to(device_in), y_in.to(device_in)
        y_hat_in = net_in(x_in)
        same_num = same_number(y_hat_in, y_in)
        metric.add(same_num, y_in.numel())
    return metric[0] / metric[1]


def train_net(net_in, train_data_in, test_data_in, num_epochs_in, lr_in, device_in):
    """
    使用trainData训练网络
    :param net_in: 输入网络
    :param train_data_in: 训练数据加载器
    :param test_data_in: 测试数据加载器
    :param num_epochs_in: 训练轮数
    :param lr_in: 学习率
    :param device_in: 设备（CPU或GPU）
    """
    def init(m):
        """
        初始化网络参数
        :param m: 网络模块
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net_in.apply(init)
    net_in.to(device_in)
    optimizer = optim.SGD(net_in.parameters(), lr=lr_in)
    loss_fun = nn.CrossEntropyLoss()
    writer = SummaryWriter(logdir='./log', comment="HDR_zyh.log")
    print('Start training on', device_in)

    for epoch in range(num_epochs_in):
        metric = Accumulator(3)
        net_in.train()
        for i, (x_in, y_in) in enumerate(train_data_in):
            x_in, y_in = x_in.to(device_in), y_in.to(device_in)
            y_hat_in = net_in(x_in)
            loss = loss_fun(y_hat_in, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * x_in.shape[0], same_number(y_hat_in, y_in), x_in.shape[0])
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = test_net(net_in, test_data_in, device_in)
        writer.add_scalar('TrainLoss', train_loss, epoch)
        writer.add_scalar('TrainAccuracy', train_acc, epoch)
        writer.add_scalar('TestAccuracy', test_acc, epoch)
        print(f'epoch {epoch+1}: train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, test acc: {test_acc:.3f}')

    # 保存模型权重
    torch.save(net_in.state_dict(), './model_weights.pth')
    print('Model weights saved to model_weights.pth')


if __name__ == "__main__":
    # 设置超参数
    net = ZyhNet()
    batch_size = 256
    trainDataIter, testDataIter = create_batch_data(batch_size)
    num_epochs = 100
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练网络
    train_net(net, trainDataIter, testDataIter, num_epochs, lr, device)

    # 加载权重文件
    model_weights = torch.load('./model_weights.pth')
    # 将权重写入到model_weights_pth.txt文件中
    with open('./model_weights_pth.txt', 'w') as f:
        for param_name, param_value in model_weights.items():
            f.write(f'Parameter name: {param_name}, Parameter value: {param_value}\n')

    # 可视化网络
    x = torch.randn(1, 1, 28, 28).to(device)
    modelData = "./model_structure.pth"
    torch.onnx.export(net, x, modelData)
    netron.start(modelData)

    # 在终端中启动tensorboard获取训练日志图
    # tensorboard serve - -logdir.\log

    print('All works done')


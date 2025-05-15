# -*- coding:utf-8 -*- #
#   name: CICV_zyh
#   time: 2025/4/12 16:53
# author: zhangyehao_202428010315039
#  email: 3074675457@qq.com
import os
import time
import datetime
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import netron
import torch.onnx

# 设置数据预处理
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪图像，裁剪后图像大小为32x32，裁剪前图像周围填充4个像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行标准化处理
])

trans_valid = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行标准化处理
])


# 创建数据加载器
def create_batch_data(batch_size_in):
    """
    创建用于训练和测试的数据加载器
    :param batch_size_in: 批次大小
    :return: 训练数据加载器和测试数据加载器
    """
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=trans_train)
    trainloader = DataLoader(trainset, batch_size=batch_size_in, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=trans_valid)
    testloader = DataLoader(testset, batch_size=batch_size_in, shuffle=False, num_workers=4)

    return trainloader, testloader


def same_number(y_hat_in, y):
    """
    返回预测值与真实值相等的个数
    :param y_hat_in: 预测值，可以是Tensor或列表
    :param y: 真实值，可以是Tensor或列表
    :return: 预测值与真实值相等的个数
    """
    y_hat_in = torch.tensor(y_hat_in) if not isinstance(y_hat_in, torch.Tensor) else y_hat_in
    y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
    if len(y_hat_in.shape) > 1 and y_hat_in.shape[1] > 1:
        y_hat_in = y_hat_in.argmax(dim=1)  # 如果预测值是多维的，获取每一行最大值的索引作为预测类别
    cmp = y_hat_in == y  # 比较预测值和真实值是否相等
    return float(cmp.sum().item())  # 返回相等的个数


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


# 定义注意力模块
class Attention(nn.Module):
    def __init__(self, dim=128, heads=8, dim_head=64, dropout=0.):
        """
        初始化注意力模块
        :param dim: 输入维度
        :param heads: 注意力头数
        :param dim_head: 每个注意力头的维度
        :param dropout: Dropout比例
        """
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要进行线性投影
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子
        self.norm = nn.LayerNorm(dim)  # 归一化层
        self.attend = nn.Softmax(dim=-1)  # Softmax层用于计算注意力权重
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 线性变换层，输出Q, K, V
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 如果需要投影则进行线性变换并添加Dropout，否则使用恒等层

    def forward(self, x_in):
        """
        前向传播
        :param x_in: 输入张量
        :return: 输出张量
        """
        x_in = self.norm(x_in)
        qkv = self.to_qkv(x_in).chunk(3, dim=-1)  # 将x_in线性变换后拆分为Q, K, V
        # 重新排列Q, K, V的维度
        q, k, v = map(lambda t_in: rearrange(t_in, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算Q和K的点积，并乘以缩放因子
        attn = self.attend(dots)  # 使用Softmax计算注意力权重
        attn = self.dropout(attn)  # 应用Dropout
        out = torch.matmul(attn, v)  # 计算注意力加权后的V
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新排列输出维度
        return self.to_out(out)  # 返回最终输出


# 定义编码器模块
class Encoder(nn.Module):
    def __init__(self, dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        """
        初始化编码器模块
        :param dim: 输入维度
        :param depth: 编码器层数
        :param heads: 注意力头数
        :param dim_head: 每个注意力头的维度
        :param mlp_dim: 前馈网络中间层维度
        :param dropout: Dropout比例
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 归一化层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 添加注意力模块
                FeedForward(dim, mlp_dim, dropout=dropout)  # 添加前馈模块
            ]))

    def forward(self, x_in):
        """
        前向传播
        :param x_in: 输入张量
        :return: 输出张量
        """
        for attn, ff in self.layers:
            x_in = attn(x_in) + x_in  # 添加残差连接
            x_in = ff(x_in) + x_in  # 添加残差连接
        return self.norm(x_in)  # 最后进行归一化


# 定义前馈模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        初始化前馈模块
        :param dim: 输入维度
        :param hidden_dim: 中间层维度
        :param dropout: Dropout比例
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 归一化层
            nn.Linear(dim, hidden_dim),  # 线性变换层，输入到中间层
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, dim),  # 线性变换层，从中间层到输出
            nn.Dropout(dropout)  # Dropout层
        )

    def forward(self, x_in):
        """
        前向传播
        :param x_in: 输入张量
        :return: 输出张量
        """
        return self.net(x_in)


# 定义ViT模型
class ViT(nn.Module):
    def __init__(self, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=512, pool='cls', channels=3, dim_head=64,
                 dropout=0.1, emb_dropout=0.1):
        """
        初始化ViT模型
        :param num_classes: 类别数
        :param dim: 输入维度
        :param depth: 编码器层数
        :param heads: 注意力头数
        :param mlp_dim: 前馈网络中间层维度
        :param pool: 池化操作类型，'cls'或'mean'
        :param channels: 图像通道数
        :param dim_head: 每个注意力头的维度
        :param dropout: Dropout比例
        :param emb_dropout: 位置嵌入的Dropout比例
        """
        super().__init__()
        image_height = 32
        patch_height = 4
        image_width = 32
        patch_width = 4
        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 图像被分成的patch数量
        patch_dim = channels * patch_height * patch_width  # 每个patch的维度
        self.to_patch_embedding = nn.Sequential(
            # 将图像重排列为patches
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # 对每个patch进行归一化
            nn.Linear(patch_dim, dim),  # 线性变换到dim维
            nn.LayerNorm(dim),  # 再次归一化
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别标记
        self.dropout = nn.Dropout(emb_dropout)  # 位置嵌入的Dropout
        self.transformer = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer编码器
        self.pool = pool
        self.to_latent = nn.Identity()  # 恒等层，用于潜在空间的提取
        self.mlp_head = nn.Linear(dim, num_classes)  # 多层感知机头部，用于分类

    def forward(self, img):
        """
        前向传播
        :param img: 输入图像
        :return: 分类结果
        """
        x_in = self.to_patch_embedding(img)  # 图像到patch嵌入
        b, n, _ = x_in.shape  # 获取batch size, patch数量和patch维度

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 复制cls_token到每个样本
        x_in = torch.cat((cls_tokens, x_in), dim=1)  # 将cls_token和patch嵌入拼接
        x_in += self.pos_embedding[:, :(n + 1)]  # 加上位置嵌入
        x_in = self.dropout(x_in)  # 应用Dropout
        x_in = self.transformer(x_in)  # 通过Transformer编码器
        x_in = x_in.mean(dim=1) if self.pool == 'mean' else x_in[:, 0]  # 根据pool类型选择输出
        x_in = self.to_latent(x_in)  # 应用潜在空间提取层
        return self.mlp_head(x_in)  # 通过多层感知机头部进行分类


# 训练模型
def train_net(net_in, traindataiter_in, testdataiter_in, num_epochs_in, lr_in, device_in):
    """
    使用trainData训练网络
    :param net_in: 输入网络
    :param traindataiter_in: 训练数据加载器
    :param testdataiter_in: 测试数据加载器
    :param num_epochs_in: 训练轮数
    :param lr_in: 学习率
    :param device_in: 设备（CPU或GPU）
    """
    net_in.to(device_in)  # 将网络移动到指定设备
    optimizer = optim.Adam(net_in.parameters(), lr=lr_in)  # 使用Adam优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    # 初始化学习率调度器（使用ReduceLROnPlateau）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    best_acc = 0.0  # 跟踪最佳准确率
    # 确保 checkpoint 目录存在
    os.makedirs('./checkpoint', exist_ok=True)
    writer = SummaryWriter(logdir='./log', comment="CICV_zyh.log")  # 创建日志写入器
    t_start = time.time()  # 记录训练开始时间
    t_start_strftime = datetime.datetime.now()
    formatted_time = t_start_strftime.strftime('%Y-%m-%d %H:%M:%S')
    print('Start training on', device_in, 'at', formatted_time)  # 打印训练开始信息
    # 开始训练
    for epoch in range(num_epochs_in):
        epoch_start_time = time.time()  # 记录每个 epoch 的开始时间
        net_in.train()  # 设置网络为训练模式
        metric = Accumulator(3)  # 用于累加训练损失、正确预测数和总样本数

        for batch_idx, (inputs, targets) in enumerate(traindataiter_in):
            inputs, targets = inputs.to(device_in), targets.to(device_in)  # 数据移动到指定设备
            optimizer.zero_grad()  # 清空梯度
            outputs = net_in(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            with torch.no_grad():
                # 累加训练信息
                metric.add(loss.item() * inputs.size(0), same_number(outputs, targets), inputs.size(0))

        train_loss = metric[0] / metric[2]  # 计算平均训练损失
        train_acc = metric[1] / metric[2]  # 计算平均训练准确率
        # 调用测试函数，获取测试损失和准确率
        test_loss, test_acc = test_net(net_in, testdataiter_in, criterion, device_in)
        # 打印当前学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        # 更新学习率（根据测试损失）
        scheduler.step(test_loss)

        # 保存最佳模型检查点
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'net': net_in.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }, './checkpoint/best_model.pth')  # 保存模型状态
            print(f'Saved best model with acc: {best_acc:.3f}')  # 打印保存信息

        writer.add_scalar('TrainLoss', train_loss, epoch)  # 写入训练损失
        writer.add_scalar('TrainAccuracy', train_acc, epoch)  # 写入训练准确率
        writer.add_scalar('TestLoss', test_loss, epoch)  # 写入测试损失
        writer.add_scalar('TestAccuracy', test_acc, epoch)  # 写入测试准确率
        writer.add_scalar('CurrentLearningRate', current_lr, epoch)  # 写入当前学习率
        print(f'Epoch {epoch + 1}: current_lr: {current_lr:.8f}, train loss: {train_loss:.4f}, '
              f'train acc: {train_acc:.4f}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')
        epoch_end_time = time.time()  # 记录每个 epoch 的结束时间
        epoch_time_elapsed = epoch_end_time - epoch_start_time  # 计算当前 epoch 耗时
        total_time_elapsed = epoch_end_time - t_start  # 计算总耗时
        current_time = datetime.datetime.now()  # 获取当前时间
        print(f'Epoch time elapsed: {epoch_time_elapsed:.2f}s, Total time elapsed: {total_time_elapsed:.2f}s, '
              f'Current time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}')
    # 保存模型权重
    torch.save(net_in.state_dict(), './model_weights.pth')  # 保存最终模型权重
    print('Model weights saved to model_weights.pth')  # 打印保存信息


# 测试模型
def test_net(net_in, testdataiter_in, criterion, device_in):
    """
    测试网络
    :param net_in: 输入网络
    :param testdataiter_in: 测试数据加载器
    :param criterion: 损失函数
    :param device_in: 设备（CPU或GPU）
    :return: 测试损失和准确率
    """
    net_in.eval()  # 设置网络为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testdataiter_in:
            inputs, targets = inputs.to(device_in), targets.to(device_in)  # 数据移动到指定设备
            outputs = net_in(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            total_loss += loss.item() * targets.size(0)  # 累加损失
            correct += same_number(outputs, targets)  # 累加正确预测数
            total += targets.size(0)  # 累加样本数

    test_loss = total_loss / total  # 计算平均测试损失
    test_acc = correct / total  # 计算平均测试准确率
    return test_loss, test_acc


# 主函数
if __name__ == "__main__":
    # 设置超参数
    net = ViT()  # 创建ViT模型实例
    batch_size = 256  # 设置批次大小
    trainDataIter, testDataIter = create_batch_data(batch_size)  # 创建数据加载器
    num_epochs = 100  # 设置训练轮数
    lr = 0.0003  # 设置学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备

    # 训练网络
    train_net(net, trainDataIter, testDataIter, num_epochs, lr, device)  # 开始训练

    # 加载权重文件
    model_weights = torch.load('./model_weights.pth')  # 加载最终模型权重
    # 将权重写入到model_weights_pth.txt文件中
    with open('./model_weights_pth.txt', 'w') as f:
        for param_name, param_value in model_weights.items():
            f.write(f'Parameter name: {param_name}, Parameter value: {param_value}\n')  # 写入权重信息

    # 可视化网络
    x = torch.randn(1, 3, 32, 32).to(device)  # 创建随机输入图像
    model_data = "./model_structure.pth"
    torch.onnx.export(net, x, model_data)  # 导出模型为ONNX格式
    ### 此处使用Netron可视化模型结构，由于结果不好保存，可减小训练轮数为2查看
    netron.start(model_data)

    # 在终端中启动tensorboard获取训练日志图
    # tensorboard serve --logdir ./log

    print('All works done')  # 打印完成信息

# -*- coding:utf-8 -*- #
#   name: AWP
#   time: 2025/4/27 12:43
# author: zhangyehao
#  email: 3074675457@qq.com
import os
import time
import datetime
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from pypinyin import pinyin, Style
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter


class BasicModule(nn.Module):
    """封装nn.Module，提供模型加载和保存接口"""

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """加载指定路径的模型"""
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """保存训练的模型到指定路径"""
        if name is None:
            pre_path = f'models/{self.model_name}_'
            name = time.strftime(f'{pre_path}%m%d_%H_%M.pth')
        os.makedirs(os.path.dirname(name), exist_ok=True)
        torch.save(self.state_dict(), name)
        print("saved model path:", name)
        return name


class PoetryModel(BasicModule):
    """自定义循环神经网络，包含embedding、LSTM、全连接层"""

    def __init__(self, vocab_size_in, embedding_dim_in, hidden_dim_in):
        super().__init__()
        self.model_name = 'PoetryModel'
        self.hidden_dim = hidden_dim_in
        self.embeddings = nn.Embedding(vocab_size_in, embedding_dim_in)
        self.lstm = nn.LSTM(embedding_dim_in, self.hidden_dim, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, vocab_size_in)
        )

    def forward(self, inputs, hidden=None):
        seq_len, batch_size_in = inputs.size()
        if hidden is None:
            h_0 = inputs.data.new(3, batch_size_in, self.hidden_dim).fill_(0).float()
            c_0 = inputs.data.new(3, batch_size_in, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(inputs)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output.view(seq_len * batch_size_in, -1))
        return output, hidden


class Accumulator:
    """构建n列变量用于累加计算指标"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PoetryDataset(Dataset):
    """自定义诗歌数据集处理类"""
    def __init__(self, data_tensor):
        # 在初始化时直接转换为LongTensor
        self.data = data_tensor.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_rhyme(char):
    """增强版韵母提取"""
    try:
        py = pinyin(char, style=Style.FINALS_TONE3, strict=False)[0][0]
        # 合并相近韵母
        replacements = {
            'ing': 'in', 'eng': 'en', 'iong': 'ong',
            'iang': 'ang', 'v': 'u'
        }
        for k, v in replacements.items():
            py = py.replace(k, v)
        return py.rstrip('012345')  # 去除声调
    except (IndexError, KeyError):
        return ''


def load_poetry_data(filename_in, batch_size_in):
    """从npz文件加载诗歌数据并创建符合规范的数据加载器"""
    dataset = np.load(filename_in, allow_pickle=True)
    # 转换为LongTensor确保数据类型正确
    data_tensor = torch.from_numpy(dataset['data']).long()
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()

    poetry_dataset = PoetryDataset(data_tensor)
    dataloader = DataLoader(
        poetry_dataset,
        batch_size=batch_size_in,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 提升GPU传输效率
    )
    return dataloader, ix2word, word2ix


def train_model(model_in, filename_in, batch_size_in, lr_in, epochs_in, device_in,
                train_writer_in, pre_model_path=None):
    """训练模型主函数"""
    global train_loss
    if pre_model_path:
        model_in.load(pre_model_path)
    model_in.to(device_in)

    dataloader, ix2word, word2ix = load_poetry_data(filename_in, batch_size_in)
    total_batches = len(dataloader)
    print(f"total numbers of batches: {total_batches}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_in.parameters(), lr=lr_in)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5)

    metric = Accumulator(2)
    t_start = time.time()  # 记录训练开始时间
    t_start_strftime = datetime.datetime.now()
    formatted_time = t_start_strftime.strftime('%Y-%m-%d %H:%M:%S')
    print('Start training on', device_in, 'at', formatted_time)  # 打印训练开始信息

    for epoch in range(epochs_in):
        for i, batch_data in enumerate(dataloader):
            batch_start_time = time.time()
            data = batch_data.transpose(1, 0).contiguous().to(device_in)
            inputs, target = data[:-1, :], data[1:, :]

            output, _ = model_in(inputs)
            loss = criterion(output, target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss * data.shape[0], data.shape[0])

            train_loss = metric[0] / metric[1]
            train_writer_in.add_scalar('Train Loss', train_loss, epoch * len(dataloader) + i)
            current_lr = scheduler.get_last_lr()[0]

            if i % 100 == 0:
                print(f'epoch: {epoch}, batch: {i}, Train Loss: {train_loss:.4f}, Learning Rate: {current_lr:.8f}')

                batch_end_time = time.time()  # 记录每个 epoch 的结束时间
                batch_time_elapsed = batch_end_time - batch_start_time  # 计算当前 epoch 耗时
                total_time_elapsed = batch_end_time - t_start  # 计算总耗时
                current_time = datetime.datetime.now()  # 获取当前时间
                print(f'Batch time elapsed: {batch_time_elapsed:.2f}s, Total time elapsed: {total_time_elapsed:.2f}s, '
                      f'Current time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}')

        scheduler.step(train_loss)

    train_writer_in.close()
    model_in.save()


def generate_poetry(model_in, filename_in, device_in, start_words, max_gen_len,
                    rhyme_pattern=None, prefix_words=None, temperature=0.8,
                    coherence_window=4, theme_keywords=None, tone='neutral'):
    """
    生成诗歌主函数

    参数说明：
    :param model_in: 训练好的模型对象，用于生成诗歌
    :param filename_in: 输入数据集的名称
    :param device_in: 设备类型，如 'cpu' 或 'cuda'
    :param start_words: 诗歌生成的起始词
    :param max_gen_len: 最大生成长度
    :param rhyme_pattern: 韵脚模式（可选）
    :param prefix_words: 前缀词列表（可选）
    :param temperature: 生成温度 (0.1~1.5)，值越小结果越确定
    :param coherence_window: 上下文连贯窗口大小 (3~6)
    :param theme_keywords: 主题关键词列表，如 ["战争","将军"]
    :param tone: 情感基调 ('positive'/'negative'/'neutral')
    """
    global w
    _, ix2word, word2ix = load_poetry_data(filename_in, 1)
    model_in.to(device_in)
    results = list(start_words)
    start_word_len = len(start_words)

    # 押韵控制初始化
    rhyme_target = None
    rhyme_positions = set()
    rhyme_indices = []
    if rhyme_pattern:
        rhyme_target = rhyme_pattern['rhyme']
        rhyme_positions = set(rhyme_pattern['positions'])
        rhyme_indices = [idx for idx, word in ix2word.items() if get_rhyme(word) == rhyme_target]

    # 状态跟踪器
    input_tensor = torch.tensor([word2ix['<START>']], device=device_in).view(1, 1).long()
    hidden = None
    used_rhyme_chars = set()
    used_words = set(start_words)
    context_window = []
    step = 0  # 有效生成步数计数器

    # 情感词库（可扩展）
    POSITIVE_WORDS = {'欢', '喜', '乐', '庆', '美', '笑', '欣', '愉'}
    NEGATIVE_WORDS = {'愁', '悲', '苦', '泪', '痛', '泣', '哀', '伤'}

    # 温度参数校验
    temperature = max(0.1, min(temperature, 2.0))

    while step < max_gen_len:
        output, hidden = model_in(input_tensor, hidden)
        logits = output / temperature  # 应用温度参数

        # ==== 生成逻辑分叉点 ====
        if step < start_word_len:
            # 1. 处理预设开头
            w = results[step]
            input_tensor = torch.tensor([word2ix[w]], device=device_in).view(1, 1)
        else:
            # ==== 概率调整模块 ====
            probs = torch.softmax(logits, dim=1)

            # 2.1 押韵位置特殊处理
            if rhyme_pattern and (step in rhyme_positions):
                current_rhyme_indices = [idx for idx in rhyme_indices if ix2word[idx] not in used_rhyme_chars]

                if not current_rhyme_indices:
                    current_rhyme_indices = rhyme_indices
                    print(f'[Warning] Rhyme pool exhausted for "{rhyme_target}"')

                # 创建押韵掩码
                rhyme_mask = torch.ones_like(logits) * -1e9
                rhyme_mask[:, current_rhyme_indices] = 0
                adjusted_logits = logits + rhyme_mask.to(device_in)
                probs = torch.softmax(adjusted_logits, dim=1)

                # 记录用过的韵脚
                used_rhyme_chars.add(w)

            # 2.2 全局重复惩罚
            for idx in range(probs.size(1)):
                word = ix2word[idx]
                repeat_penalty = 0.3 if word in used_words else 1.0
                if word in context_window[-7:]:  # 近期重复加强惩罚
                    repeat_penalty *= 0.5
                probs[0][idx] *= repeat_penalty

            # 2.3 主题增强
            if theme_keywords:
                for idx in range(probs.size(1)):
                    word = ix2word[idx]
                    if word in theme_keywords:
                        probs[0][idx] *= 2  # 主题词概率增强

            # 2.4 情感一致性维护
            if tone != 'neutral':
                target_words = POSITIVE_WORDS if tone == 'positive' else NEGATIVE_WORDS
                opposite_words = NEGATIVE_WORDS if tone == 'positive' else POSITIVE_WORDS
                for idx in range(probs.size(1)):
                    word = ix2word[idx]
                    if word in target_words:
                        probs[0][idx] *= 1.2
                    elif word in opposite_words:
                        probs[0][idx] *= 0.3

            # ==== 采样模块 ====
            probs = probs / probs.sum()  # 重新归一化
            top_index = torch.multinomial(probs, 1).item()
            w = ix2word[top_index]

            # ==== 状态更新 ====
            input_tensor = torch.tensor([top_index], device=device_in).view(1, 1)
            results.append(w)
            used_words.add(w)
            context_window.append(w)
            if len(context_window) > coherence_window:
                context_window.pop(0)

        # ==== 终止条件判断 ====
        if w == '<EOP>':
            del results[-1]
            break

        step += 1  # 步进计数

    return results


def generate_acrostic(model_in, filename_in, device_in, start_words_acrostic, max_gen_len,
                      rhyme_pattern=None, prefix_words=None, temperature=0.8,
                      coherence_window=4, theme_keywords=None, tone='neutral'):
    """生成藏头诗主函数，集成诗歌生成功能

    参数说明：
    :param model_in: 训练好的模型对象
    :param filename_in: 数据集文件名
    :param device_in: 计算设备 ('cpu'或'cuda')
    :param start_words_acrostic: 藏头字列表
    :param max_gen_len: 最大生成长度
    :param rhyme_pattern: 押韵模式字典 {positions: set(), rhyme: str}
    :param prefix_words: 前缀词列表（预留参数）
    :param temperature: 温度参数 (0.1~2.0)
    :param coherence_window: 上下文连贯窗口
    :param theme_keywords: 主题关键词列表
    :param tone: 情感基调 ('positive'/'negative'/'neutral')
    """
    # 数据加载与初始化
    _, ix2word, word2ix = load_poetry_data(filename_in, 1)
    model_in.to(device_in)

    # 初始化生成状态
    results = []
    index = 0  # 藏头字指针
    pre_word = '<START>'
    input_tensor = torch.tensor([word2ix['<START>']], device=device_in).view(1, 1).long()
    hidden = None
    step = 0  # 总生成步数计数器

    # 押韵控制初始化
    rhyme_target = None
    rhyme_positions = set()
    rhyme_indices = []
    if rhyme_pattern:
        rhyme_target = rhyme_pattern['rhyme']
        rhyme_positions = set(rhyme_pattern['positions'])
        rhyme_indices = [idx for idx, word in ix2word.items() if get_rhyme(word) == rhyme_target]

    # 上下文管理
    used_words = set()
    context_window = []
    used_rhyme_chars = set()

    # 情感词库
    POSITIVE_WORDS = {'欢', '喜', '乐', '庆', '美', '笑', '欣', '愉'}
    NEGATIVE_WORDS = {'愁', '悲', '苦', '泪', '痛', '泣', '哀', '伤'}

    # 参数校验
    temperature = max(0.1, min(temperature, 2.0))

    while step < max_gen_len:
        output, hidden = model_in(input_tensor, hidden)

        # ==== 藏头字插入逻辑 ====
        if pre_word in {'。', '！', '<START>'} and index < len(start_words_acrostic):
            # 强制插入藏头字
            w_in = start_words_acrostic[index]
            index += 1

            # 更新输入状态
            input_tensor = torch.tensor([word2ix[w_in]], device=device_in).view(1, 1)
            results.append(w_in)
            pre_word = w_in

            # 更新上下文状态
            used_words.add(w_in)
            context_window.append(w_in)
            if len(context_window) > coherence_window:
                context_window.pop(0)

            step += 1
            continue

        # ==== 正常生成逻辑 ====
        logits = output / temperature

        # 押韵调整
        if rhyme_pattern and (step in rhyme_positions):
            current_rhyme_indices = [idx for idx in rhyme_indices if ix2word[idx] not in used_rhyme_chars]

            if not current_rhyme_indices:  # 韵脚池耗尽处理
                current_rhyme_indices = rhyme_indices
                print(f'[Warning] Rhyme pool exhausted for "{rhyme_target}"')

            # 创建押韵掩码
            rhyme_mask = torch.ones_like(logits) * -1e9
            rhyme_mask[:, current_rhyme_indices] = 0
            logits = logits + rhyme_mask.to(device_in)

        # 生成概率处理
        probs = torch.softmax(logits, dim=1)

        # 重复惩罚机制
        for idx in range(probs.size(1)):
            word = ix2word[idx]
            repeat_penalty = 0.3 if word in used_words else 1.0
            if word in context_window[-coherence_window:]:
                repeat_penalty *= 0.5  # 加强近期重复惩罚
            probs[0][idx] *= repeat_penalty

        # 主题增强
        if theme_keywords:
            for idx in range(probs.size(1)):
                word = ix2word[idx]
                if word in theme_keywords:
                    probs[0][idx] *= 2

        # 情感一致性调整
        if tone != 'neutral':
            target_words = POSITIVE_WORDS if tone == 'positive' else NEGATIVE_WORDS
            opposite_words = NEGATIVE_WORDS if tone == 'positive' else POSITIVE_WORDS
            for idx in range(probs.size(1)):
                word = ix2word[idx]
                if word in target_words:
                    probs[0][idx] *= 1.2
                elif word in opposite_words:
                    probs[0][idx] *= 0.3

        # 概率归一化
        probs = probs / probs.sum()

        # 采样下一个词
        top_index = torch.multinomial(probs, 1).item()
        w_in = ix2word[top_index]

        # 终止条件判断
        if w_in == '<EOP>':
            break

        # 更新生成状态
        results.append(w_in)
        input_tensor = torch.tensor([top_index], device=device_in).view(1, 1)
        pre_word = w_in
        used_words.add(w_in)
        context_window.append(w_in)
        if len(context_window) > coherence_window:
            context_window.pop(0)

        # 记录用过的韵脚
        if rhyme_pattern and (step in rhyme_positions):
            used_rhyme_chars.add(w_in)

        step += 1

    return results





if __name__ == "__main__":
    # 配置参数
    train_loss = float('inf')
    filename = 'data/tangshi_data.npz'
    batch_size = 32
    lr = 0.001
    epochs = 10
    vocab_size = 8293
    embedding_dim = 128
    hidden_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练示例
    """
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    vis_dir = time.strftime('assets/' + model.model_name + '_%m%d_%H_%M')
    train_writer = SummaryWriter(f'{vis_dir}/Train')
    train_model(model, filename, batch_size, lr, epochs, device, train_writer)
    """

    # 生成示例
    """
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    model.load('models/PoetryModel_0428_07_26.pth')
    # 生成七言绝句，押an韵，max_gen_len包含标点符号
    generated = generate_poetry(
        model, filename, device,
        start_words='秦时明月汉时关',
        max_gen_len=32,
        rhyme_pattern={
            'rhyme': 'an',
            'positions': [14, 22, 30]
        },
        temperature=0.3,
        theme_keywords=["战争"],
        tone='negative'
    )
    poetry = ''.join(generated)
    print('\n'.join(poetry.split('。')))
    """

    # 藏头诗示例（带押韵和主题），max_gen_len包含标点符号

    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    model.load('models/PoetryModel_0428_07_26.pth')
    acrostic =  generate_acrostic(model, filename, device,
                                  start_words_acrostic=['春','风','得','意'], max_gen_len=48,
                                  rhyme_pattern={'positions': {10,22,34,46}, 'rhyme': 'ang'},
                                  theme_keywords=['季节', '自然'], temperature=0.7, tone='positive')
    print('\n'.join(''.join(acrostic).split('。')))



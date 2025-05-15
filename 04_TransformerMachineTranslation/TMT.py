# -*- coding:utf-8 -*- #
#   name: TMT
#   time: 2025/5/14 01:21
# author: zhangyehao
#  email: 3074675457@qq.com
# 标准库导入
import math
import os
import re
import random
import time
import copy
import logging
import numpy as np
# 第三方库导入
import jieba
import nltk
import sacrebleu
import torch
import torch.nn.functional as fuc
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# 下载包的必要数据
nltk.download('punkt_tab')
nltk.download('words')


class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        """
        初始化
        :param src: 源数据(经tokenizer处理后)
        :param tgt: 目标数据(经tokenizer处理后)
        """
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]

    def __len__(self):
        return len(self.src)


class Tokenizer:
    def __init__(self, en_path_in, ch_path_in, count_min=5):
        """
        初始化
        :param en_path_in: 英文数据路径
        :param ch_path_in: 中文数据路径
        :param count_min: 对出现次数少于这个次数的数据进行过滤
        """
        self.en_path = en_path_in
        self.ch_path = ch_path_in
        self.__count_min = count_min
        self.en_data = self.__read_ori_data(en_path_in)
        self.ch_data = self.__read_ori_data(ch_path_in)
        self.index_2_word = ['unK', '<pad>', '<bos>', '<eos>']
        self.word_2_index = {'unK': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        self.en_set = set()
        self.en_count = {}
        self.__en_count = {}
        self.__ch_count = {}
        self.__count_word()
        self.mx_length = 40
        self.data_ = []
        self.__filter_data()
        random.shuffle(self.data_)
        self.test = self.data_[-1000:]
        self.data_ = self.data_[:-1000]

    def __read_ori_data(self, path):
        """
        读取原始数据
        :param path: 数据路径
        :return: 返回一个列表，每个元素是一条数据
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __count_word(self):
        """
        统计中英文词汇表
        """
        logging.info("One minute... building a Chinese-English vocabulary list.")
        for data in self.en_data:
            sentence = word_tokenize(data)
            for sen in sentence:
                if sen in self.en_set:
                    self.en_count[sen] += 1
                else:
                    self.en_set.add(sen)
                    self.en_count[sen] = 1
        for k, v in self.en_count.items():
            if v >= self.__count_min:
                self.word_2_index[k] = len(self.index_2_word)
                self.index_2_word.append(k)
            else:
                self.word_2_index[k] = 0
        self.en_set = set()
        self.en_count = {}
        for data in self.ch_data:
            sentence = list(jieba.cut(data))
            for sen in sentence:
                if sen in self.en_set:
                    self.en_count[sen] += 1
                else:
                    self.en_set.add(sen)
                    self.en_count[sen] = 1
        for k, v in self.en_count.items():
            if v >= self.__count_min:
                self.word_2_index[k] = len(self.index_2_word)
                self.index_2_word.append(k)
            else:
                self.word_2_index[k] = 0

    def __filter_data(self):
        length = len(self.en_data)
        for i in range(length):
            self.data_.append([self.en_data[i], self.ch_data[i], 0])
            self.data_.append([self.ch_data[i], self.en_data[i], 1])

    def en_cut(self, data):
        data = word_tokenize(data)
        if len(data) > self.mx_length:
            return 0, []
        en_tokens = []
        for tk in data:
            en_tokens.append(self.word_2_index.get(tk, 0))
        return 1, en_tokens

    def ch_cut(self, data):
        data = list(jieba.cut(data))
        if len(data) > self.mx_length:
            return 0, []
        en_tokens = []
        for tk in data:
            en_tokens.append(self.word_2_index.get(tk, 0))
        return 1, en_tokens

    def encode_all(self, data):
        """
        对一组数据进行编码
        :param data: 一个数组，形状为n*3 每个元素是[src_sentence, tgt_sentence, label]
        :return:
        """
        src = []
        tgt = []
        en_src, en_tgt, l = [], [], []
        labels = []
        for i in data:
            en_src.append(i[0])
            en_tgt.append(i[1])
            l.append(i[2])
        for i in range(len(l)):
            if l[i] == 0:
                lab1, src_tokens = self.en_cut(en_src[i])
                if lab1 == 0:
                    continue
                lab2, tgt_tokens = self.ch_cut(en_tgt[i])
                if lab2 == 0:
                    continue
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                labels.append(i)
            else:
                lab1, tgt_tokens = self.en_cut(en_tgt[i])
                if lab1 == 0:
                    continue
                lab2, src_tokens = self.ch_cut(en_src[i])
                if lab2 == 0:
                    continue
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                labels.append(i)
        return labels, src, tgt

    def encode(self, src, l):
        if l == 0:
            src1 = word_tokenize(src)
            en_tokens = []
            for tk in src1:
                en_tokens.append(self.word_2_index.get(tk, 0))
            return [en_tokens]
        else:
            src1 = list(jieba.cut(src))
            en_tokens = []
            for tk in src1:
                en_tokens.append(self.word_2_index.get(tk, 0))
            return [en_tokens]

    def decode(self, data):
        """
        数据解码
        :param data: 这里传入一个中文的index
        :return: 返回解码后的一个字符
        """
        return self.index_2_word[data]

    def __get_datasets(self, data):
        """
        获取数据集
        :return:返回DataSet类型的数据 或者 None
        """
        labels, src, tgt = self.encode_all(data)
        return TranslationDataset(src, tgt)

    def another_process(self, batch_datas):
        """
        特殊处理，这里传入一个batch的数据，并对这个batch的数据进行填充，使得每一行的数据长度相同。这里填充pad 空字符  bos 开始  eos结束
        :param batch_datas: 一个batch的数据
        :return: 返回填充后的数据
        """
        en_index, ch_index = [], []
        en_len, ch_len = [], []

        for en, ch in batch_datas:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        max_en_len = max(en_len)
        max_ch_len = max(ch_len)
        max_len = max(max_en_len, max_ch_len + 2)

        en_index = [i + [self.word_2_index['<pad>']] * (max_len - len(i)) for i in en_index]
        ch_index = [[self.word_2_index['<bos>']] + i + [self.word_2_index['<eos>']] +
                    [self.word_2_index['<pad>']] * (max_len - len(i) + 1) for i in ch_index]

        en_index = torch.tensor(en_index)
        ch_index = torch.tensor(ch_index)
        return en_index, ch_index

    def get_dataloader(self, data, batch_size=40):
        """
        得到dataloader
        """
        data = self.__get_datasets(data)
        return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.another_process)

    def get_vocab_size(self):
        return len(self.index_2_word)

    def get_dataset_size(self):
        return len(self.en_data)


def subsequent_mask(size):
    """
    注意力机制掩码生成
    :param size: 句子长度
    :return: 注意力掩码
    """
    attn_shape = (1, size, size)
    subsequent_mask_in = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_in) == 0


class Batch:
    def __init__(self, src, trg=None, tokenizer_in=None, device='cuda'):
        """
        初始化函数
        :param src: 源数据
        :param trg: 目标数据
        :param tokenizer_in: 分词器
        :param device: 训练设备
        """
        src = src.to(device).long()
        trg = trg.to(device).long()
        self.src = src
        self.__pad = tokenizer_in.word_2_index['<pad>']
        self.src_mask = (src != self.__pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, : -1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, self.__pad)
            self.ntokens = (self.trg_y != self.__pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        生成掩码矩阵
        :param tgt: 目标数据
        :param pad: 填充字符的索引
        :return:
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        """
        词嵌入层初始化
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        """
        位置编码器层初始化
        :param d_model: 词嵌入维度
        :param dropout: dropout比例
        :param max_len: 序列最大长度
        :param device: 训练设备
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0.0, max_len, device=device)
        position.unsqueeze_(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        pe[:, 0:: 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1:: 2] = torch.cos(torch.mm(position, div_term))
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        多头注意力机制初始化
        :param h: 多头
        :param d_model: 词嵌入维度
        :param dropout: dropout比例
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        反向传播
        :param query: q
        :param key: k
        :param value: v
        :param mask: 掩码
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        注意力加权
        :param query: q
        :param key: k
        :param value: v
        :param mask: 掩码矩阵
        :param dropout: dropout比例
        :return:
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = fuc.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        """
        子层连接结构初始化层
        :param d_model: 词嵌入纬度
        :param dropout: dropout比例
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        return x + x_


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈全连接网络初始化层
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = fuc.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        """
        编码器层初始化
        :param h: 头数
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x_in: self.self_attn(x_in, x_in, x_in, mask))
        return self.norm(self.sublayer2(x, self.feed_forward))


class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        """
        解码器层
        :param h: 头数
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(Decoder, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer1(x, lambda x_in: self.self_attn(x_in, x_in, x_in, tgt_mask))
        x = self.sublayer2(x, lambda x_in: self.src_attn(x_in, m, m, src_mask))
        return self.norm(self.sublayer3(x, self.feed_forward))


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        """
        生成器层初始化
        :param d_model:
        :param vocab:
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return fuc.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, tokenizer_in, h=8, d_model=256, E_N=2, D_N=2, device='cuda'):
        """
        transformer层初始化
        :param h: 头数
        :param d_model: 词嵌入纬度
        :param tokenizer_in:
        :param E_N:
        :param D_N:
        :param device:
        """
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([Encoder(h, d_model) for _ in range(E_N)])
        self.decoder = nn.ModuleList([Decoder(h, d_model) for _ in range(D_N)])
        self.src_embed = Embedding(d_model, tokenizer_in.get_vocab_size())
        self.tgt_embed = Embedding(d_model, tokenizer_in.get_vocab_size())
        self.src_pos = PositionalEncoding(d_model, device=device)
        self.tgt_pos = PositionalEncoding(d_model, device=device)
        self.generator = Generator(d_model, tokenizer_in.get_vocab_size())

    def encode(self, src, src_mask):
        """
        编码
        :param src: 源数据
        :param src_mask: 源数据掩码
        :return:
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        for i in self.encoder:
            src = i(src, src_mask)
        return src

    def decode(self, memory, tgt, src_mask, tgt_mask):
        """
        解码
        :param memory: 编码器输出
        :param tgt: 目标数据输入
        :param src_mask: 源数据掩码
        :param tgt_mask: 目标数据掩码
        :return:
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        for i in self.decoder:
            tgt = i(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        反向传播
        :param src: 源数据
        :param tgt: 目标数据
        :param src_mask: 源数据掩码
        :param tgt_mask: 目标数据掩码
        :return:
        """
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        初始化
        :param size: 目标数据词表大小
        :param padding_idx: 目标数据填充字符的索引
        :param smoothing: 做平滑的值，为0即不进行平滑
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        反向传播
        :param x: 预测值
        :param target: 目标值
        :return:
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        """
        初始化
        :param generator: 生成器
        :param opt: 经wormup后的optimizer
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        类做函数调用
        :param x: 经transformer解码后的结果
        :param y: 目标值
        :param norm: 本次数据有效的字符数，即，除去padding后的字符数
        :return:
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化
        :param model_size: 词嵌入维度
        :param factor:
        :param warmup:
        :param optimizer:
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def is_english_sentence(sentence):
    """
    检查句子是否为英文句子
    """
    english_pattern = re.compile(r'[a-zA-Z]')
    match = english_pattern.search(sentence)
    if match:
        return True
    else:
        return False


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, Variable(ys), src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, i])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def compute_bleu4(tokenizer_in, random_integers, model, device):
    """
    计算BLEU4
    :param tokenizer_in: tokenizer
    :param random_integers: 随机测试集数据编号
    :param model: 模型
    :param device: 设备
    :return:
    """
    m1, m2 = [], []
    m3, m4 = [], []
    model.eval()
    da = []
    for i in random_integers:
        da.append(tokenizer_in.test[i])
    labels, x, _ = tokenizer_in.encode_all(da)
    with torch.no_grad():
        y = predict(x, model, tokenizer_in, device)
    p = 0
    itg = []
    if len(y) != 100:
        return 0
    for i in labels:
        itg.append(random_integers[i])
    for i in itg:
        if is_english_sentence(tokenizer_in.test[i][1]):
            m1.append(tokenizer_in.test[i][1])
            m2.append([y[p]])
        else:
            m3.append(list(jieba.cut(tokenizer_in.test[i][1])))
            m4.append([list(jieba.cut(y[p]))])
        p += 1
    smooth = SmoothingFunction().method1
    b1 = [sacrebleu.sentence_bleu(candidate, refs).score for candidate, refs in zip(m1, m2)]
    for i in range(len(m4)):
        b2 = sentence_bleu(m4[i], m3[i], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100
        b1.append(b2)
    return sum(b1) / len(b1)


# 训练
def train(epochs_num=30, batch_size=40):
    device = 'cuda'
    model = Transformer(tokenizer, device=device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(device)
    criteria = LabelSmoothing(tokenizer.get_vocab_size(), tokenizer.word_2_index['<pad>'])
    optimizer = NoamOpt(256, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    lossF = SimpleLossCompute(model.generator, criteria, optimizer)
    epochs = epochs_num
    model.train()
    loss_all = []
    logging.info(f'vocab_size: {tokenizer.get_vocab_size()}')
    if not os.path.exists('./model'):
        os.makedirs('./model')
    t = time.time()
    data_loader = tokenizer.get_dataloader(tokenizer.data_, batch_size=batch_size)
    batchs = []
    for index, data in enumerate(data_loader):
        src, tgt = data
        batch = Batch(src, tgt, tokenizer_in=tokenizer, device=device)
        batchs.append(batch)
    best_bleu = 0
    for epoch in range(epochs):
        p = 1
        for batch in batchs:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            out = lossF(out, batch.trg_y, batch.ntokens)
            if p % 1000 == 0:
                model.eval()
                logging.info(f'epoch: {epoch} |batch: {p//1000} |loss: {float(out / batch.ntokens)}')
                model.train()
                logging.info(f'time elapsed(s): {time.time() - t}')
                if float(out / batch.ntokens) < 2.5:
                    random_integers = random.sample(range(len(tokenizer.test)), 100)
                    bleu = compute_bleu4(tokenizer, random_integers, model, device)
                    logging.info(f'bleu4: {bleu}')
                    if (bleu > best_bleu) & (bleu > 14):  # 如果当前的BLEU分数大于最佳的BLEU分数
                        best_bleu = bleu  # 更新最佳的BLEU分数
                        torch.save(model.state_dict(), f'./model/translation_best_bleu_{bleu}.pt')  # 保存模型
                        logging.info(f'New best model saved with BLEU score: {bleu}')
            p += 1
        loss_all.append(float(out / batch.ntokens))
    with open('./model/loss.txt', 'w', encoding='utf-8') as f:
        f.write(str(loss_all))


# 预测
def predict(data, model, tokenizer_in, device='cuda'):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    with torch.no_grad():
        data1 = []
        for i in range(len(data)):
            src = torch.from_numpy(np.array(data[i])).long().to(device)
            src = src.unsqueeze(0)
            src_mask = (src != tokenizer_in.word_2_index['<pad>']).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len=100, start_symbol=tokenizer_in.word_2_index['<bos>'])
            translation = []
            for j in range(1, out.size(1)):
                sym = tokenizer_in.index_2_word[out[0, j].item()]
                if sym != '<eos>':
                    translation.append(sym)
                else:
                    break
            if len(translation) > 0:
                if translation[0].lower() in words.words():
                    data1.append(TreebankWordDetokenizer().detokenize(translation))
                else:
                    data1.append("".join(translation))
        return data1


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(
        r"C:\Users\zhangyehao\Desktop\2025courses\03_Deeplearning\homework\DeepLearningHomework\04_TransformerMachineTranslation\data")
    # 配置 logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    en_path = r"C:\Users\zhangyehao\Desktop\2025courses\03_Deeplearning\homework\DeepLearningHomework\04_TransformerMachineTranslation\data\src.txt"
    ch_path = r'C:\Users\zhangyehao\Desktop\2025courses\03_Deeplearning\homework\DeepLearningHomework\04_TransformerMachineTranslation\data\tgt.txt'
    tokenizer = Tokenizer(en_path, ch_path, count_min=3)
    train(epochs_num=30, batch_size=40)


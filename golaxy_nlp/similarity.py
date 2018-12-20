#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/6 10:20
"""
import os
import sys
import numpy as np
from gensim.models import Word2Vec
import time
import pickle
import random
def getPackagePath():
    path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(path))
sys.path.append(getPackagePath())
from golaxy_nlp.dataload import clean_sentence

def _load_tfidf():
    """
    加载tfidf模型
    :return:
    """
    with open("./model/tfidf","rb") as f:
        tfidf = pickle.load(f)
    return tfidf

def _load_w2v():
    """
    加载word2vec模型
    :return:
    """
    w2v = Word2Vec.load('./model/commentsvec')
    return w2v

class WMD():
    """
    计算WMD距离，主要实现了WCD方法
    """
    def __init__(self,w2v):
        """
        计算词向量矩阵和词典
        :param w2v:
        """
        self.w2v = w2v
        self.X,self.vocab = self.getWeigthMatrix()

    def getWeigthMatrix(self):
        """
        获取词向量矩阵及词典
        :param w2v:
        :return:
        """
        vocabulary = self.w2v.wv.vocab
        X = np.zeros((self.w2v.layer1_size,len(vocabulary)),dtype=np.float64)
        for key in vocabulary.keys():
            value = vocabulary.get(key)
            index = value.index
            X[:,index] = self.w2v.wv[key]
        return X,vocabulary

    def transform_to_vec(self,words={}):
        """
        转化词组为向量形式
        :param vocabulary:
        :param words:
        :return:
        """
        result = np.zeros(len(self.w2v.wv.vocab),dtype=np.float64)
        for word in words.keys():
            value = self.w2v.wv.vocab.get(word)
            if value is not None:
                index = value.index
                result[index] = words.get(word)
        return result

    def _compute_tf(self,w):
        """
        计算词频MAP
        :param w:
        :return:
        """
        wordList = w.split(" ")
        all_count = len(wordList) * 1.0
        wordMap = {}
        for word in wordList:
            if wordMap.__contains__(word):
                wordMap[word] += 1
            else:
                wordMap[word] = 1
        for word in wordMap.keys():
            wordMap[word] = wordMap.get(word)/all_count
        return wordMap

    def similarity_wcd(self,w1,w2):
        """
        计算WMD距离下界---WCD距离
        :param w1:
        :param w2:
        :return:
        """
        words1 = self._compute_tf(w1)
        words2 = self._compute_tf(w2)
        v1 = self.transform_to_vec(words1)
        v2 = self.transform_to_vec(words2)
        return eculidean(self.w2v.wv.syn0.transpose().dot(v1),self.w2v.wv.syn0.transpose().dot(v2))

def cosine(a,b):
    """
    余弦相似度计算
    :param a:
    :param b:
    :return:
    """
    M = np.sum(a * b)
    Z = np.linalg.norm(a) * np.linalg.norm(b)
    result = float("%.2f" % np.abs(M/Z))
    return result

def eculidean(a,b):
    """
    欧氏距离 2范数
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a-b,2)

def _kl(p,q):
    """
    KL散度 相对熵
    :param p:
    :param q:
    :return:
    """
    p += 0.000001
    q += 0.000001
    c = p/q
    t = p * np.log(c)
    return np.sum(t)

def KL(p,q):
    """
    平均KL散度
    :param p:
    :param q:
    :return:
    """
    s1 = _kl(p,q)
    s2 = _kl(q,p)
    return (s1+s2)/2

def word2vec_transform(w2v,sentence):
    """
    word2vec 转化句子为向量
    :param w2v:
    :param sentence:
    :return:
    """
    size = w2v.layer1_size
    data = sentence.split(" ")
    length = len(data)
    vec = np.zeros(shape=(1,size),dtype=np.float32)
    for word in data:
        try:
            vec += w2v.wv[word]
        except:
            length -= 1
            continue
    vec = vec/length
    return vec

def max_pooling(v,vec):
    """
    对向量做最大池化
    :param v:
    :param vec:
    :return:
    """
    length = vec.shape
    for i in range(length[0]):
        if vec.data[i] < v.data[i]:
            vec.data[i] = v.data[i]

def window_sampling(vecs,window=3):
    N = len(vecs)
    window_vecs = []
    iter = N - window + 1
    for i in range(iter):
        mean_vec = 0
        for j in range(window):
            mean_vec += vecs[i + j]
        mean_vec = mean_vec / window
        window_vecs.append(mean_vec)
    return window_vecs

def word2vec_transform_hierachical(w2v,sentence):
    vecs = []
    size = w2v.layer1_size
    data = sentence.split(" ")
    length = len(data)
    for word in data:
        try:
            v = w2v.wv[word]
            vecs.append(v)
        except Exception as e:
            length -= 1
    v = window_sampling(vecs,window=3)
    return v

def word2vec_transform_maxpooling(w2v,sentence):
    size = w2v.layer1_size
    data = sentence.split(" ")
    length = len(data)
    vec = np.zeros(shape=(1, size), dtype=np.float32)
    for word in data:
        try:
            v = w2v.wv[word]
            max_pooling(v,vec)
        except:
            length -= 1
    return vec

def tfidf_transform(tfidf,sentence):
    """
    tfidf转化为词袋向量
    :param tfidf:
    :param sentence:
    :return:
    """
    s = tfidf.transform([sentence])
    return s.toarray()

def get_sentences():
    """
    读取相应格式的句子
    :return:
    """
    r = []
    count = 0
    with open("./data/label_file.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            count += 1
            if count == 1 : continue
            line = line.replace("\n","")
            data = line.split(",")
            score = data[0]
            t1 = data[1]
            t2 = data[2]
            p1 = clean_sentence(t1)
            p2 = clean_sentence(t2)
            r.append((p1,p2,t1,t2,score))
    return r

def test():
    tfidf = _load_tfidf()
    # w2v = _load_w2v()
    # wmd = WMD(w2v)
    # sentence_pairs = get_sentences()
    sentence_pairs=[("违法 通报 鸿茅药酒 保健品","鸿茅药酒 跌落 神坛 如何 合理 质疑 鸿茅药酒"),
                    ("中国海军 演练 内容 充满 悬念 南海 动作","自媒体 诽谤 伊利 批捕 伊利 董事长 纯属 捏造 恶意 诽谤"),
                    ("蔡英文 登舰 出海 舰长 透露 接获 命令","外媒 辽宁舰 南海军演 美国 航母 现身 中国周边"),
                    ("博鳌 亚洲论坛 2018年 年会 海南 博鳌 举行","王毅 博鳌 亚洲论坛 2018年 年会 亚洲 世界 博鳌智慧 贡献 博鳌力量 "),
                    ("青岛 上合组织 峰会 时间 上合组织 峰会 几天","王毅 推动 青岛峰会 实现 目标 上合组织 新时代"),
                    ("中兴 内部人士 我们 甚至 禁止 高通 英特尔 打电话","普京 宣誓 就职 外交部 这样 祝贺"),
                    ("腾讯 没有 梦想 其实 腾讯 梦想 一般人 看不懂","马化腾 深夜 梦想 刷屏 腾讯 公关 总监 假的")]
    for k,pair in enumerate(sentence_pairs):
        p1 = pair[0]
        p2 = pair[1]
        # title_a = pair[2]
        # title_b = pair[3]
        # score = pair[4]
        t1 = tfidf_transform(tfidf,p1)
        t2 = tfidf_transform(tfidf, p2)
        # w1 = word2vec_transform(w2v,p1)
        # w2 = word2vec_transform(w2v,p2)
        print("Pair: ",k)
        # print("Title: ",title_a,"\t",title_b,"\t score: ",score)
        tfidf_value = cosine(t1,t2)
        # s = float(score)
        print("Tfidf sim: %s " % (tfidf_value))
        # print("Word2vec sim: %s" % (cosine(w1,w2)))
        # print("Word2vec wmd sim: %s" %(wmd.similarity_wcd(p1,p2)))

def save_idf():
    """
    存储idf词典
    :return:
    """
    tfidf = _load_tfidf()
    vocab = tfidf.vocabulary_
    idf = tfidf._tfidf.idf_
    with open("./files/idf.utf8","a",encoding="utf-8") as f:
        for word in vocab:
            index = vocab.get(word)
            score = idf[index]
            f.write(str(word)+"\t"+str(score)+"\n")
# test()
def load_tfidf():
    """

    :return:
    """
    with open("./files/idf.utf8", "r", encoding="utf-8")as file:
        result = {}
        lines = file.readlines()
        for line in lines:
            line = line.replace("\n", "")
            split_data = line.split('\t')
            key = split_data[0]
            value = float(split_data[1])
            result[key] = value
        return result
idf_map = load_tfidf()
#
def extract_words_with_idf(words,number):
    words_split = words.split(' ')
    result_dict = {}
    count = 0
    for word in words_split:
        if(idf_map.__contains__(word)):
            count += 1
            if result_dict.__contains__(word):
                tf = result_dict[word][1]
                result_dict[word] = (idf_map[word],tf+1)
            else:
                result_dict[word] = (idf_map[word],1)
    for key,value in result_dict.items():
        result_dict[key] = value[0]*(1.0* value[1] / count)
    sort_dict = sorted(result_dict.items(), key=lambda x:x[1], reverse=True)
    result = []
    for sort_tuple in sort_dict[:number]:
        result.append(sort_tuple[0])
    return ' '.join(result)
# content = "美国 制裁 上瘾 左右为难 竟是 国家 来源 直通车 紧张不安 美国 伊朗 俄罗斯 土耳其 发动 制裁 受此 影响 俄罗斯卢布 周一 跌至 年初 最低 土耳其 里拉 周一 开盘 崩跌 南非兰特 新兴 市场 货币 大跌 印尼盾 亚洲 表现 最差 货币 糟糕 伊朗 特朗普 退出 伊核 协议 不到 三个 伊朗 货币 大幅 贬值 里亚尔 美元 贬值 近半 伊朗 仍称 近期 美国 谈判 土耳其 俄罗斯 政要 美国 开打 经济战 值得注意 面对 美国 制裁 印度 左右为难 直通车 侯雨彤 制图 印度 感到 为难 印度 感到 为难 美国 伊朗 制裁 美国 总统 特朗普 退出 伊核 协议 签署 执行 备忘录 美国政府 为期 两个 阶段 伊朗 执行 相关 制裁 第一阶段 制裁 能源 领域 制裁 刚刚开始 特朗普 退出 伊核 协议 伊朗 货币 大幅 贬值 特朗普 曾多次 放软 身段 谈判 伊朗 态度强硬 松口 外媒 报道 伊朗 领袖 哈梅内伊 德黑兰 华盛顿 谈判 伊朗 总统 鲁哈尼 早前 敌人 刺伤 某人 希望 谈谈 第一件 扔掉 值得注意 特朗普 社交 媒体 伊朗 商业 往来 美国 商业 交往 制裁 提升 水平 美国 伊朗 制裁 瞄准 伊朗 石油 天然气 行业 美国 世界 石油 进口国 伊朗 石油 进口量 消减 没能 国家 面临 美国 连坐 制裁 面对 美国 压力 欧洲 做出 强硬 抵制 启用 阻断 保护 伊朗 境内 运营 欧盟 企业 免受 美国 制裁 日本 美国 谈判 韩国 断绝 伊朗 进口 石油 相比之下 印度 纠结 上半年 印度 伊朗 进口 原油 日均 万桶 伊朗 原油 进口国 伊朗 出口量 印度 美国 出口 贸易额 伊朗 印度 美国 外交关系 处于 上升期 印度 难以 取舍 分析 指出 印度 最终 很大 程度 取决于 大国 美国 制裁 抵制 力度 印度 特朗普 一份 豁免权 侥幸心理 新兴 市场 扛得住 美国 制裁 影响 土耳其 里拉 危机 蔓延 新兴 市场 扩散 数据 显示 MSCI 国际 新兴 市场 货币 指数 下跌 一年 最低水平 MSCI 日本 亚太地区 股票指数 下跌 分析 人士 指出 土耳其 危机 美元 强令 亚洲 市场 亚洲 货币 面临 进一步 震荡 风险 值得注意 土耳其 MSCI 国际 新兴 市场 指数 中仅 摩根 大通 新兴 市场 债券"
# print(extract_words_with_idf(content,10))
#
#
# from jieba.analyse import TextRank
# t = TextRank()
# print(t.textrank(content,topK=10))
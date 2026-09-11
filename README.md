# Bag of Words Meets Bags of Popcorn —— 电影评论情感分析

一个基于 Kaggle 经典入门竞赛 **["Bag of Words Meets Bags of Popcorn"](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/part-2-word-vectors)** 的 NLP 学习项目。目标是对 IMDb 电影评论做**情感二分类**（正面 / 负面）。

项目严格跟随官方教程的三个部分，从零实现了 **词袋模型 → Word2Vec → KMeans 聚类** 三种由浅入深的文本向量化方法，并用随机森林做分类。

> 学习笔记：本项目的完整教程地址为
> <https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/part-2-word-vectors>

---

## 目录

- [项目背景](#项目背景)
- [三种方法](#三种方法)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [运行顺序](#运行顺序)
- [结果](#结果)
- [踩坑记录](#踩坑记录)
- [参考](#参考)

---

## 项目背景

竞赛提供的数据集为 IMDb 电影评论：

| 数据集 | 数量 | 说明 |
|--------|------|------|
| `labeledTrainData.tsv` | 25,000 条 | 带情感标签（1=正面，0=负面）的训练集 |
| `testData.tsv` | 25,000 条 | 无标签的测试集 |
| `unlabeledTrainData.tsv` | 50,000 条 | 无标签，用于训练 Word2Vec 词向量 |

原始评论文本需要经过清洗：去除 HTML 标签、URL、非字母字符，转小写，剔除停用词。

---

## 三种方法

### 1. 词袋模型（Bag of Words）+ 随机森林
> 脚本：`TrainRandomForest.py`

- 用 `sklearn` 的 `CountVectorizer` 将评论文本转为词频向量（`max_features=5000`）
- 用 `RandomForestClassifier`（100 棵树）分类
- 这是 NLP 最基础的特征表示方法

### 2. Word2Vec + 平均段落向量
> 脚本：`TrainWord2VecModel.py` → `AverageParagraphVector.py`

- 用 `gensim` 在 75,000 条评论（训练集 + 无标签集）上训练 Word2Vec 词向量
- 词向量参数：`vector_size=300`、`min_count=40`、`window=10`、`sample=1e-3`
- 每条评论中所有词的向量**取平均**，得到定长的段落向量（300 维）
- 再送入随机森林分类

### 3. Word2Vec + KMeans 聚类（Bag of Centroids）
> 脚本：`TrainWord2VecModel.py` → `Clustering.py`

- 对训练好的词向量做 KMeans 聚类（簇数 = 词表大小 / 5 ≈ 3271）
- 每条评论统计每个簇中出现的词频，得到"簇词袋"向量
- 再送入随机森林分类

---

## 项目结构

```
Bag_of_Words/
├── Code/                         # 源代码
│   ├── CookData.py               # 数据清洗工具函数
│   ├── ModelParameters.py        # 模型超参数
│   ├── TrainRandomForest.py      # 方法一：词袋模型 + 随机森林
│   ├── TrainWord2VecModel.py     # 训练 Word2Vec 词向量
│   ├── Word2VecTest.py           # 测试 Word2Vec 模型
│   ├── AverageParagraphVector.py # 方法二：平均段落向量 + 分类
│   └── Clustering.py             # 方法三：KMeans 聚类 + 分类
├── Data/                         # 数据集（从 Kaggle 下载，不随仓库分发）
├── Model/                        # 训练出的 Word2Vec 模型（生成产物）
├── Result/                       # 分类结果 CSV（生成产物）
├── requirements.txt              # 依赖清单
├── .gitignore
└── README.md
```

---

## 环境要求

| 项 | 要求 |
|----|------|
| **Python** | **3.12**（⚠️ 不能用 3.13/3.14，gensim 4.4.0 没有对应预编译 wheel，会从源码编译失败） |
| 依赖包 | 见 `requirements.txt` |
| NLTK 数据 | `stopwords`、`punkt`、`punkt_tab` |

核心依赖版本：

| 包 | 版本 | 用途 |
|----|------|------|
| pandas | 3.0.5 | 读写 TSV/CSV |
| beautifulsoup4 | 4.15.0 | 剥离 HTML 标签 |
| nltk | 3.10.3 | 分句、停用词 |
| scikit-learn | 1.9.0 | CountVectorizer、KMeans、随机森林 |
| gensim | 4.4.0 | Word2Vec 词向量 |

---

## 快速开始

```powershell
# 1. 克隆仓库
git clone <你的仓库地址>
cd Bag_of_Words

# 2. 用 Python 3.12 创建虚拟环境
py -3.12 -m venv .venv
.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载 NLTK 数据
python -m nltk.downloader stopwords punkt punkt_tab

# 5. 从 Kaggle 下载数据并放入 Data/ 目录
#    竞赛页面: https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data
#    需要: labeledTrainData.tsv / testData.tsv / unlabeledTrainData.tsv
```

---

## 运行顺序

```powershell
# 方法一：词袋模型
python Code/TrainRandomForest.py
# 输出: Result/Bag_of_Words_model.csv

# 先训练 Word2Vec 模型（方法二、三共用）
python Code/TrainWord2VecModel.py
# 输出: Model/300features_40minwords_10context_Word2VecModel

# 可选：测试词向量质量
python Code/Word2VecTest.py

# 方法二：平均段落向量
python Code/AverageParagraphVector.py
# 输出: Result/Word2Vec_AverageVectors.csv

# 方法三：KMeans 聚类
python Code/Clustering.py
# 输出: Result/BagOfCentroids.csv
```

---

## 结果

三种方法在 25,000 条测试评论上的预测结果输出到 `Result/` 目录：

| 方法 | 输出文件 |
|------|----------|
| 词袋模型 + 随机森林 | `Bag_of_Words_model.csv` |
| Word2Vec 平均向量 + 随机森林 | `Word2Vec_AverageVectors.csv` |
| Word2Vec 聚类 + 随机森林 | `BagOfCentroids.csv` |

---

## 踩坑记录

本项目基于 2014 年的老教程，代码中的 API 已随 Python / gensim 版本更新而废弃，以下改动已同步进代码：

1. **`dict.keys()` / `dict.values()` 不能下标索引**（Python 3 返回视图对象）→ 先 `list(...)` 转换。
2. **gensim 4.x 参数改名**：`size` → `vector_size`；`syn0` → `vectors`；`init_sims()` 已废弃可删除。
3. **`counter += 1.` 会让计数器变成浮点数**，导致 numpy 数组下标索引报 `IndexError`，应写成 `counter += 1`。
4. **Python 3.14 与 gensim 不兼容**：gensim 4.4.0 尚无 cp314 预编译 wheel，需使用 Python 3.12。

---

## 参考

- [Kaggle 竞赛官方教程：Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/part-2-word-vectors)
- [gensim 官方文档](https://radimrehurek.com/gensim/)
- [scikit-learn 官方文档](https://scikit-learn.org/)

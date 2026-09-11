import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import re
from nltk.corpus import stopwords
import numpy as np

# 我们只是用 BeautifulSoup 剥离 HTML 标签，输入文本本身可能含 URL，
# 因此过滤掉它对"像 URL 的输入"的误报警告。
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

stops = set(stopwords.words("english"))

#服务于词袋模型的数据清洗
def review_to_words( raw_review ):
    #去除HTML元素
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    #去除除a-z A-Z之外的符号
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    #全改为小写字母并拆分
    words = letters_only.lower().split()

    #剔除停止词
    meaningful_words = [w for w in words if w not in stops]

    return( " ".join( meaningful_words ))

"""
输入:
"I love this movie. It is really great. The acting is superb."

得到:
    'love movie really great acting superb'

"""

#服务于Word2Vec模型,最终将某句话拆成某句话的单词列表
def review_to_wordlist(review, remove_stopwords = False):
    #去除HTML元素
    review_text = BeautifulSoup(review, "html.parser").get_text()

    #清除url链接(带协议的、www开头的、以及裸域名)
    review_text = re.sub(r'(?:https?://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?', '', review_text, flags=re.IGNORECASE)

    #去除除a-z A-Z之外的符号
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    #全改为小写字母并拆分
    words = review_text.lower().split()

    #剔除停止词(可选)
    if remove_stopwords:
        #stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return (words)

"""
输入:
    "i love this movie."
    "It is really great."
    "The acting is superb."

得到:
    "i", "love", "this", "movie"
    "it", "is", "really", "great"
    "the", "acting", "is", "superb"

将句子拆成单词列表
"""

#使用 NLTK 的 punkt 分词器进行句子分割，得到句子列表
def review_to_sentences(review, tokenizer, remove_stopwords = False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences
"""
输入:
"I love this movie. It is really great. The acting is superb."

得到:
[
    ["i love this movie."],
    ["It is really great."],
    ["The acting is superb."]
]

将段落拆成句子
"""


#使用word2vec模型对段落中所有词的词向量进行向量求和，最后平均得到平均段落向量
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype = "float32")
    nwords = 0.
    #将modle中的词制作成一个词表，用来加速查询
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model.wv[word])
    # 防止某条评论里所有词都不在词表时出现除零错误
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

#批量处理某个段落的所有词汇
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = "float32")
    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

#获取簇编号计数向量
def create_bag_of_centroids(wordlist, word_centroid_map):
    #找出最大簇号并且+1
    num_centroids = max(word_centroid_map.values()) +1
    bag_of_centroids = np.zeros(num_centroids, dtype = "float32")
    #统计词出现的簇类编号
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids
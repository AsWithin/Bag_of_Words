import pandas as pd
import os
import CookData as CData
import nltk.data
import logging
from gensim.models import word2vec
from ModelParameters import WORD2VEC_PARAMS

#拼接所需数据path
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "..", "Data", "labeledTrainData.tsv")
test_path = os.path.join(current_dir, "..", "Data", "testData.tsv")
unlabeled_path = os.path.join(current_dir, "..", "Data", "unlabeledTrainData.tsv")
model_dir = os.path.join(current_dir, "..", "Model")

#读取数据
train = pd.read_csv(train_path, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(test_path, header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv(unlabeled_path, header=0, delimiter="\t", quoting=3)

#测试数据是否读取正确
#print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n"
#       % (train["review"].size,  test["review"].size, unlabeled_train["review"].size))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

#清洗数据
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += CData.review_to_sentences(review, tokenizer)

#清洗数据
print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += CData.review_to_sentences(review, tokenizer)

# print (len(sentences))
# print (sentences[0])
# print (sentences[1])

#开启日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" #模型参数
num_features = 300      #词向量维度
min_word_count = 40      #最小词频
num_workers = 8         #并行线程数
context = 10            #上下文窗口
downsampling = 1e-3     #高频词下采样

#初始化并且开始训练模型
print("Training model...")
model = word2vec.Word2Vec(
    sentences,
    workers = num_workers,
    vector_size = num_features,
    min_count = min_word_count,
    window = context,
    sample = downsampling,
    epochs = 15
    ) """

print("Training model...")
model = word2vec.Word2Vec(sentences, **WORD2VEC_PARAMS)

model_name = os.path.join(model_dir, "300features_40minwords_10context_Word2VecModel")
model.save(model_name)
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
import CookData as CData
from ModelParameters import RANDOM_FOREST_PARAMS

#测试停止词表是否正确
#print(stopwords.words("english"))

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "Data", "labeledTrainData.tsv")

train = pd.read_csv(data_path , header=0, delimiter='\t', quoting=3)
#绝对路径寻址
#train = pd.read_csv("../Data/labeledTrainData.tsv" , header=0, delimiter='\t', quoting=3)


#数据集输出测试
#print (train["review"][0])

#example = BeautifulSoup(train["review"][0])

#输出去除HTML标签的数据集测试
#print(example.get_text())

#letters_only = re.sub("[^a-zA-Z]", " ", example.get_text())

#输出去除除a-z A-Z之外的符号的数据集输出测试
#print(letters_only)

#lower_case = letters_only.lower()
#words = lower_case.split()

#words = [w for w in words if not w in stopwords.words("english")]
#print (words)

#加载停止词表
"""
stops = set(stopwords.words("english"))

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

#测试函数是否正确运行
#clean_review = review_to_words(train["review"][0])
#print(clean_review)

#获取数据集大小
num_reviews = train["review"].size

#初始化干净训练评论数组，准备接收处理后的数据
clean_train_reviews = []

print ("Cleaning and parsing the training set movie reviews...\n")

for i in range(0 , num_reviews):
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append(CData.review_to_words(train["review"][i]))

#print (clean_train_reviews[1])

#创建词袋模型，按单词拆分，只保留频率最高的5000个单词，不过滤停用词
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

#根据训练数据训练出向量长度为5000的计数向量
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

#print (train_data_features.shape)

#vocab = vectorizer.get_feature_names_out()
#print(vocab)

#dist = np.sum(train_data_features, axis=0)
#for tag, count in zip(vocab, dist):
#    print(count , tag)

#训练随机森林模型
print ("Training the random forest...")
# forest = RandomForestClassifier(n_estimators= 100, verbose=2)
forest = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
forest = forest.fit(train_data_features, train["sentiment"])

#获取测试集数据
data_path = os.path.join(current_dir, "..", "Data", "testData.tsv")
test = pd.read_csv(data_path, header=0, delimiter="\t", quoting=3)

print(test.shape)
num_reviews = len(test["review"])
clean_test_reviews = []

#清洗测试集数据
print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if((i+1) % 1000 == 0):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = CData.review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#使用模型进行预测
result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output_path = os.path.join(current_dir, "..", "Result", "Bag_of_Words_model.csv")

#将结果写入Bag_of_Words_model.csv
output.to_csv(output_path, index=False, quoting=3)
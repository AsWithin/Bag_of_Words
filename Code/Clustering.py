from sklearn.cluster import KMeans
import time
import pandas as pd
import os
from gensim.models import Word2Vec
import CookData as CData
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ModelParameters import RANDOM_FOREST_PARAMS

current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "..", "Data", "labeledTrainData.tsv")
test_path = os.path.join(current_dir, "..", "Data", "testData.tsv")
model_path = os.path.join(current_dir, "..", "Model", "300features_40minwords_10context_Word2VecModel")

#加载模型
model = Word2Vec.load(model_path)

#加载数据
train = pd.read_csv(train_path, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(test_path, header=0, delimiter="\t", quoting=3)

#清洗数据，并且剔除停止词
print("清洗训练集数据")
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(CData.review_to_wordlist(review, remove_stopwords = True))

#清洗测试集数据
print("清洗测试集数据")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(CData.review_to_wordlist(review, remove_stopwords = True))

#开始计时
start = time.time()

word_vectors = model.wv.vectors
num_clusters = word_vectors.shape[0] // 5 #集群数量

kmens_clustering = KMeans(n_clusters = num_clusters)
idx = kmens_clustering.fit_predict(word_vectors)

end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

#将模型中的词表和聚类的编号对应合在一起，制作成字典
word_centroid_map = dict(zip(model.wv.index_to_key, idx))

# keys = list(word_centroid_map.keys())
# values = list(word_centroid_map.values())

# for cluster in range(0,10):
#     print("\nCluster %d" % cluster)
#     words = []
#     for i in range(len(values)):
#         if values[i] == cluster:
#             words.append(keys[i])
#     print(words)

#创建训练簇计数向量
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = CData.create_bag_of_centroids(review, word_centroid_map)
    counter += 1

#创建测试簇计数向量
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = CData.create_bag_of_centroids(review, word_centroid_map)
    counter += 1

#创建并训练随机森林分类器
forest = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

#输出分类结果
output_path = os.path.join(current_dir, "..", "Result", "BagOfCentroids.csv")
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv(output_path, index = False, quoting = 3)
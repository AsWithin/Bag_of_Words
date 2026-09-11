from gensim.models import Word2Vec
import os
import pandas as pd
import CookData as CData
from ModelParameters import WORD2VEC_PARAMS, RANDOM_FOREST_PARAMS
from sklearn.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "..", "Data", "labeledTrainData.tsv")
test_path = os.path.join(current_dir, "..", "Data", "testData.tsv")
model_path = os.path.join(current_dir, "..", "Model", "300features_40minwords_10context_Word2VecModel")

# 加载模型
model = Word2Vec.load(model_path)

type(model.wv.vectors)
model.wv.vectors.shape

#加载训练数据
train = pd.read_csv(train_path, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(test_path, header=0, delimiter="\t", quoting=3)

#清洗数据，并且剔除停止词
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(CData.review_to_wordlist(review, remove_stopwords = True))

num_features = WORD2VEC_PARAMS["vector_size"]

#得到平均段落向量
print("Creating average feature vecs for test reviews")
trainDataVecs = CData.getAvgFeatureVecs(clean_train_reviews, model, num_features)

#清洗测试集数据
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(CData.review_to_wordlist(review, remove_stopwords = True))

#得到平均段落向量
testDataVecs = CData.getAvgFeatureVecs(clean_test_reviews, model, num_features)

#初始化随机森林分类器
forest = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

print("Fitting a random forest to labeled training data...")
#训练随机森林
forest = forest.fit(trainDataVecs, train["sentiment"])

result = forest.predict(testDataVecs)

#输出分类结果
output_path = os.path.join(current_dir, "..", "Result", "Word2Vec_AverageVectors.csv")
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv(output_path, index = False, quoting = 3)
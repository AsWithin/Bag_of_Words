# 这是一个用于测试生成出来的 Word2Vec 模型的代码
from gensim.models import Word2Vec
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "Model", "300features_40minwords_10context_Word2VecModel")

# 加载模型
model = Word2Vec.load(model_path)

print("=" * 60)
print(f"模型已加载：词表大小 = {len(model.wv)}，向量维度 = {model.wv.vector_size}")
print("=" * 60)


def show_vector(word):
    """安全地展示某个词的向量：词名和向量一起打印出来。"""
    if word not in model.wv:
        print(f"[警告] 词 '{word}' 不在词表内（可能被 min_count 过滤掉了）")
        return
    vector = model.wv[word]
    print(f"\n词 '{word}' 对应的向量（shape = {vector.shape}）：")
    print(vector)


# 1. 查看词向量（词名 + 向量一起展示）
for w in ["review", "movie", "good"]:
    show_vector(w)


# 2. 两个词之间的相似度（越接近 1 越相似）
print("\n" + "=" * 60)
print("两个词之间的相似度：")
print(f"  man  vs woman : {model.wv.similarity('man', 'woman'):.4f}")
print(f"  good vs great : {model.wv.similarity('good', 'great'):.4f}")
print(f"  good vs bad   : {model.wv.similarity('good', 'bad'):.4f}")


# 3. 不相似匹配测试：找出一组词里最格格不入的那个
print("\n" + "=" * 60)
print("不相似匹配测试（找出最不像的词）：")
NoMatchTest = [
    "man woman child kitchen",
    "france england germany berlin",
    "paris berlin london austria",
]
for test in NoMatchTest:
    result = model.wv.doesnt_match(test.split())
    print(f"  '{test}'  ->  最不像的是：{result}")


# 4. 相似匹配测试：找出与目标词最相近的词
print("\n" + "=" * 60)
print("相似匹配测试（与目标词最相近的词）：")
for word in ["man", "queen", "awful"]:
    print(f"\n  与 '{word}' 最相似的 5 个词：")
    for similar_word, score in model.wv.most_similar(word, topn=5):
        print(f"    {similar_word:<15s} {score:.4f}")


# 5. 词类比测试：king - man + woman ≈ queen
print("\n" + "=" * 60)
print("词类比测试（king - man + woman）：")
for word, score in model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=5):
    print(f"    {word:<15s} {score:.4f}")

#模型参数

#随机森林参数
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,    #树的数量
    "verbose": 2            #控制训练过程中日志输出详细程度
                            #verbose=0	静默模式，什么都不打印（默认值）
                            #verbose=1	显示基本进度信息
                            #verbose=2	显示详细的训练过程信息，比如每棵树的构建进度
}

#Word2Vec参数
WORD2VEC_PARAMS = {
    "vector_size": 300,     #词向量维度
    "window": 10,           #上下文窗口大小
    "min_count": 40,        #最低词频
    "workers": 8,           #并行线程数
    "epochs": 15,           #训练轮数
    "sample": 1e-3          #高频词下采样
}
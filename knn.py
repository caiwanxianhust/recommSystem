import numpy as np


class KNNBasic(object):
    def __init__(self, k):
        self.k = k
        self.mean = None
        self.std = None
        self.X = None
        self.n_classes = None
        self.labels = None
        self.label2id = None
        self.id2label = None
        self.y_onehot = None
        self.sim = None
        self.index_sorted = None

    def fit(self, X, y):
        """
                :param x: ndarray, (n_samples, n_features)
                :param y: iterable, (n_samples,)
                :return: object of KNNClassifier
                """
        # 标准化
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.X = (X - self.mean) / self.std
        # self.convertLabel2Vec(y)

    def predict(self, x):
        raise NotImplementedError

    def sortNeighbor(self, x):
        x = (x - self.mean) / self.std
        # 计算相似度 (n_pre, n_samples)
        self.sim = np.matmul(x, self.X.T)
        # 排序，降序
        self.index_sorted = np.argsort(self.sim, axis=1)[:, ::-1]

    def convertLabel2Vec(self, y):
        self.label2id = {}
        self.n_classes = 0
        for label in y:
            if label not in self.label2id:
                self.label2id[label] = self.n_classes
                self.n_classes += 1
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.labels = np.array([self.label2id[_] for _ in y])
        self.y_onehot = np.eye(self.n_classes)[self.labels]


class KNNClassifier(KNNBasic):
    def fit(self, X, y):
        super(KNNClassifier, self).fit(X, y)
        self.convertLabel2Vec(y)

    def predict(self, x):
        """
        :param x: ndarray, (n_pre, n_features)
        :return:
        """
        # self.convertLabel2Vec(y)
        assert self.X is not None, "must fit first!"
        x = np.array(x, dtype=np.float32)
        x = x.reshape(-1, self.X.shape[1])
        self.sortNeighbor(x)
        # 统计前K个邻居，投票决定预测结果
        y_pred = np.zeros((x.shape[0], self.k, self.n_classes),
                          dtype=np.float32)
        for i in range(x.shape[0]):
            for j in range(self.k):
                y_pred[i, j, :] = self.y_onehot[self.index_sorted[i, j], :]
        y_pred = np.sum(y_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return [self.id2label[_] for _ in y_pred]


class KNNRegressor(KNNBasic):
    def fit(self, X, y):
        super(KNNRegressor, self).fit(X, y)
        self.labels = np.array(y, dtype=np.float32).reshape(len(y), )

    def predict(self, x):
        assert self.X is not None, "must fit first!"
        x = np.array(x, dtype=np.float32)
        x = x.reshape(-1, self.X.shape[1])
        self.sortNeighbor(x)
        y_pred = np.zeros((x.shape[0],), dtype=np.float32)
        for i in range(x.shape[0]):
            for j in range(self.k):
                y_pred[i] += self.labels[j]
            y_pred[i] /= self.k
        return y_pred


if __name__ == "__main__":
    features = np.array([[180, 76], [158, 43], [176, 78], [161, 49]])
    # labels = ["男", "女", "男", "女"]
    labels = [0, 1, 0, 1]
    # knn = KNNClassifier(3)
    knn = KNNRegressor(3)
    knn.fit(features, labels)
    # print(knn.label2id, knn.id2label)
    # print(knn.labels)
    # print(knn.y_onehot)
    # print(knn.n_classes)
    # print((knn.index_sorted))
    x_test = np.array([176, 76])
    y_pred = knn.predict(x_test)
    print(y_pred)

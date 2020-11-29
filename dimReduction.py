import abc
import numpy as np


class DimReducBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, x):
        pass

    def transform(self, x):
        pass

    def fit_transform(self, x):
        pass


class PCA(DimReducBase):
    def __init__(self, k, **kwargs):
        self.k = k
        self.f_val = None
        self.f_vec = None

    def fit(self, x):
        """
        :param x: ndarray, (n_samples, n_features)
        :return: object of PCA
        """
        # 去中心化
        x = x - np.mean(x, axis=0)
        # 协方差矩阵，（n_features, n_features）
        cov_matrix = np.cov(x, rowvar=False)
        # 特征值特征矩阵，（n_features,）(n_features, n_features)
        f_val, f_vec = np.linalg.eig(cov_matrix)
        # 得到特征值的排序索引，降序
        val_index_sorted = np.argsort(f_val)[::-1]
        # 重排特征值，降序(n_features)
        self.f_val = f_val[val_index_sorted]
        # 重排特征向量，降序(n_features, n_features)
        self.f_vec = f_vec[:, val_index_sorted]
        return self

    def transform(self, x):
        """
        :param x: ndarray, (n_samples, n_features)
        :return: object of PCA
        """
        assert (self.f_val is not None and self.f_vec is not None), "must fit first!"
        return np.matmul(x, self.f_vec[:, :self.k])

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


if __name__ == "__main__":
    pca = PCA(k=2)
    a = np.arange(12).reshape(3, 4)
    # pca.fit(a)
    # new_a = pca.transform(a)
    new_a = pca.fit(a).transform(a)
    print(new_a)
    print(pca.f_val)
    b = np.arange(28).reshape(7, 4)
    new_b = pca.transform(b)
    print(new_b)

import numpy as np


class BayesBasic(object):
    def __init__(self):
        pass


    def fit(self, X, y):
        pass


    def predict(self, x):
        pass


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



class GaussNaiveBayesClassifier(BayesBasic):
    def __init__(self):
        self.classP = {}
        self.classP_features = dict()

    def fit(self, X, y):
        self.convertLabel2Vec(y)
        for idx in self.labels:
            self.classP[idx] = self.classP.get(idx, 0) + 1 / len(y)
        for c in self.id2label:
            self.classP_features[c] = {}
            for i in range(X.shape[1]):
                feature = X[np.equal(self.labels, c)][:, i]
                mean = np.mean(feature, axis=0)
                std = np.std(feature, axis=0)
                self.classP_features[c][i] = {"mean": mean, "std": std}

    def gaussian(self, mu, sigma, x):
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    def predict(self, x):
        assert len(self.classP) > 0, "must fit first!"
        x = np.array(x, dtype=np.float32)
        print(x)
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        y_pred = np.zeros((x.shape[0], self.n_classes), dtype=np.float32)
        for labelId, label_p in self.classP.items():
            current_p = np.ones((x.shape[0], 1), dtype=np.float32) * label_p
            for featureId, mean_std in self.classP_features[labelId].items():
                current_p *= self.gaussian(mean_std["mean"], mean_std["std"], x[:, featureId])
            y_pred[:, labelId] = current_p
        self.y_pred = y_pred
        return [self.id2label[_] for _ in np.argmax(y_pred, axis=1)]





if __name__ == "__main__":
    gaussBayes = GaussNaiveBayesClassifier()
    data = np.array([
        [320, 204, 198, 265],
        [253, 53, 15, 2243],
        [53, 32, 5, 325],
        [63, 50, 42, 98],
        [1302, 523, 202, 5430],
        [32, 22, 5, 143],
        [105, 85, 70, 322],
        [872, 730, 840, 2762],
        [16, 15, 13, 52],
        [92, 70, 21, 693]
    ])
    labels = ["t", "f", "f", "t", "f", "f", "t", "t", "t", "f"]
    gaussBayes.fit(data, labels)
    print(gaussBayes.classP)
    print(gaussBayes.classP_features)
    x = np.array([134, 84, 235, 349])
    y = gaussBayes.predict(x)
    print(y)


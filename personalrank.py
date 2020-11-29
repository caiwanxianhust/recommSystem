import heapq
import numpy as np
import time


class PersonalRank(object):
    def __init__(self, alpha, epochs, Graph):
        self.alpha = alpha
        self.epochs = epochs
        self.Graph = Graph
        self.rank = dict()

    def fit(self, root):
        """
        :param Graph: dict; {nodeA:{nodeB:weight}}
        :param root:
        :return:
        """
        rank = {k: 0 for k in self.Graph.keys()}
        rank[root] = 1
        for epoch in range(self.epochs):
            tmp = {k: 0 for k in self.Graph.keys()}
            for nodeA, ri in self.Graph.items():
                for nodeB, wij in ri.items():
                    tmp[nodeB] += self.alpha * rank[nodeA] / len(ri)
            tmp[root] += 1 - self.alpha
            rank = tmp
        self.rank[root] = rank
        return self

    def mfit(self, root):
        self.M = np.zeros(shape=(len(self.Graph), len(self.Graph)), dtype=np.float64)
        keys = list(self.Graph.keys())
        key_index = {i: keys.index(i) for i in keys}
        for nodeA, ri in self.Graph.items():
            for nodeB in ri:
                self.M[key_index[nodeA], key_index[nodeB]] = 1 / len(ri)
        r0 = np.zeros((len(self.Graph), 1), dtype=np.float64)
        r0[key_index[root], [0]] = 1.0
        r = r0
        for epoch in range(self.epochs):
            r = (1 - self.alpha) * r0 + self.alpha * np.matmul(self.M.T, r)
        self.rank[root] = {node: r[idx, 0] for node, idx in key_index.items()}
        return self

    def getRank(self, root):
        assert root in self.rank, "must fit root first!"
        if root in self.rank:
            return self.rank[root]

    def getTopN(self, n, root, removeUsed=True):
        if root not in self.rank:
            return []
        if removeUsed:
            rank = [(nodeB, value) for nodeB, value in self.rank[root].items() \
                    if (nodeB not in self.Graph[root] and nodeB != root)]
        else:
            rank = [(nodeB, value) for nodeB, value in self.rank[root].items() if nodeB != root]
        topN = heapq.nlargest(n, rank, key=lambda x: x[1])
        return topN


if __name__ == '__main__':
    alpha = 0.85
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}
    pr = PersonalRank(alpha, 100, G)
    start_time = time.time()
    pr.fit("A")
    end_time = time.time()
    print(end_time - start_time)
    # print(pr.getRank("A"))
    # print(pr.getRank("B"))
    # topN = pr.getTopN(2, "A")
    # print(topN)
    print(pr.rank["A"])
    end_time_1 = time.time()
    pr.mfit("A")
    print(end_time_1 - end_time)
    # print(pr.M)
    print(pr.rank["A"])
    # print(pr.getRank("A"))

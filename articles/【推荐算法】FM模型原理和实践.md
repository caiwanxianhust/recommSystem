## 1 算法背景
### 1.1 什么是FM模型
**FM**英文全称是“Factorization Machine”，简称FM模型，中文名“**因子分解机**”，2010年由Rendle提出，近年来在各大厂的CTR预估和推荐领域广泛使用。  
### 1.2 逻辑回归
在计算广告和推荐系统中，判断一个商品是否进行推荐需要将CTR预估(click-through rate)的点击率作为依据。早期CTR预估使用最多的就是**逻辑回归**这种“线性模型+人工特征组合引入非线性”的模式，简单且容易并行化，可以轻松处理上亿条数据，所以目前仍然有不少实际系统仍然采取这种模式。但是逻辑回归最大的缺陷就是人工特征工程，耗时费力，那么能否将特征组合的能力体现在模型层面呢？
$$
逻辑回归:\quad
\ln \frac {y}{1-y} = w_0 + \sum_{i=1}^{n} {w_i x_i}
$$
### 1.3 加入特征组合的逻辑回归
$$
加入特征组合的逻辑回归：\quad
\ln \frac {y}{1-y} = w_0 + \sum_{i=1}^{n} {w_i x_i} + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{w_{ij}x_i x_j}
$$
其实想达到这一点并不难，在上述逻辑回归公式里加入二阶特征组合即可，任意两个特征进行组合，可以将这个组合出的特征看作一个新特征，融入线性模型中，和一阶特征权重一样，这个组合特征权重在训练阶段学习获得。这种二阶特征组合的使用方式，和多项式核SVM是等价的。虽然这个模型看上去貌似解决了二阶特征组合问题了，但是它有个潜在的问题：它对组合特征建模，泛化能力比较弱，尤其是在大规模稀疏特征存在的场景下，很难直接学习到合适的权重，恰好CTR预估和推荐排序这些场景的最大特点就是特征的大规模稀疏。所以上述模型并未在工业界广泛采用。那么，有什么办法能够解决这个问题吗？ 
### 1.4 FM模型
$$
FM模型：\quad
y = w_0 + \sum_{i=1}^{n} {w_i x_i} + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_i, v_j \gt x_i x_j}
$$
其中，$v \in R^{n \times k}$，$\lt v_i, v_j \gt$表示的是两个长度为k的向量的点积。  
$$
\lt v_i, v_j \gt = \sum_{f=1}^{k}{v_{i,f} \cdot v_{j,f}}
$$
为了解决这个问题FM应运而生。FM模型也直接引入任意两个特征的二阶特征组合，与上述加入特征组合的线性模型的区别在于特征组合权重的计算方法。FM对于每个特征，学习一个大小为k的一维向量，于是，两个特征$x_i$ 和$x_j$的特征组合的权重值，通过特征对应的向量$v_i$和$v_j$的内积$\lt v_i, v_j \gt$来表示。这本质上是在对特征进行embedding化表征，和目前常见的各种实体embedding本质思想是一脉相承的，但是很明显在2010年提出FM的时候，基于深度学习的各种花里胡哨的embedding方法还不流行，所以FM作为特征embedding，可以看作当前深度学习里各种embedding方法的前辈了。  
### 1.4 基于矩阵分解的协调过滤（MF）模型和FM模型的关系
当然，FM这种模式有它的前辈模型吗？有，详见笔者的另一篇文章：[【推荐算法】基于矩阵分解的协同过滤算法](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247483693&idx=3&sn=4f4b394394a96eea34fbbeabc9b5342d&chksm=e9278494de500d82cb41bdecd4099c36ac1d66f4f2896f6143c2913e9c08f3804614956b16df&token=760429129&lang=zh_CN#rd)。MF（Matrix Factorization，矩阵分解）模型是个在推荐系统领域里经典的协同过滤模型。核心思想是通过两个低维小矩阵（一个代表用户embedding矩阵，一个代表物品embedding矩阵）的乘积计算，来模拟真实用户点击或评分产生的大的协同信息稀疏矩阵，本质上是编码了用户和物品协同信息的降维模型。使用梯度下降算法训练后，每个用户和物品得到对应的低维embedding表达，如果要预测某个用户u对商品i的评分的时候，只要它们做个内积计算 $q^T_i p_u$，这个得分就是预测得分。本质上，MF模型是FM模型的特例，MF可以被认为是只有User ID和Item ID这两个特征Fields的FM模型，MF将这两类特征通过矩阵分解，来达到将这两类特征embedding化表达的目的。而FM则可以看作是MF模型的进一步拓展，除了User ID和Item ID这两类特征外，很多其它类型的特征，都可以进一步融入FM模型里，它将所有这些特征转化为embedding低维向量表达，并计算任意两个特征embedding的内积，就是特征组合的权重。  
从谁更早使用特征embedding表达这个角度来看的话，很明显，和FM比起来，MF才是真正的前辈，无非是特征类型比较少而已。而FM继承了MF的特征embedding化表达这个优点，同时引入了更多Side information作为特征，将更多特征及Side information embedding化融入FM模型中。所以很明显FM模型更灵活，能适应更多场合的应用范围。  
因此，基于MF和FM两者的关系，可以得出下面的观点：  
其一：实际工程中可以优先考虑使用FM而非传统的MF，因为可以在实现等价功能的基础上，很方便地融入其它任意特征，充分发挥Side information的作用。  
其二：在排序阶段，只使用ID信息的模型是不实用的，原因很简单，大多数真实应用场景中，User/Item有很多信息可用，而协同数据只是其中的一种，引入更多特征明显对于更精准地进行个性化推荐是非常有帮助的。MF模型通常只是作为一路召回的形式存在。

## 2 FM模型优化
可以看出上述原始FM模型的时间复杂度是$O(k \cdot n^2)$，现实生活应用中的n往往是个非常巨大的特征数，对于一个实用化模型来说，效果是否足够好只是一个方面，计算效率是否够高也很重要，优秀的算法两者缺一不可。  
FM如今被广泛采用并成功替代LR模型的一个关键所在是：它可以通过数学公式改写，把时间复杂度从$O(k \cdot n^2)$降低到$O(k \cdot n)$，其中n是特征数量，k是特征的embedding size。简化推导过程如下：  
$$
\begin{split}
\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_i, v_j \gt x_i x_j} &= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\lt v_i, v_j \gt x_i x_j}- \frac{1}{2}\sum_{i=1}^{n}{\lt v_i, v_i \gt x_i x_i}\\ 
&= \frac{1}{2}(\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}{v_{i,f} v_{j,f} x_i x_j} - \sum_{i=1}^{n}\sum_{f=1}^{k}{v_{i,f}^2 x_i^2})\\ 
&= \frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f} x_i)(\sum_{j=1}^{n}v_{j,f} x_j) - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2})\\ 
&= \frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f} x_i)^2 - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2})
\end{split}
$$
经过这样的分解之后，我们就可以通过随机梯度下降SGD进行求解：  
$$
\frac {\partial y(X)}{\partial \theta} = 
\begin{cases}
1, & \theta = w_0 \\
x_i, & \theta = w_i \\
x_i \sum_{j=1}^{n}{v_{i,f}x_j}-v_{i,f}x_i^2, & \theta = v_{i,f}
\end{cases}
$$

## 3 基于tensorflow2.X的fm模型实践

这里使用的是Kaggle比赛——Telco Customer Churn电信客户流失数据集。

### 3.1 导入包
```python
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from fm import FM, FMLayer

# 设置GPU显存动态增长
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
```
### 3.2 fm实现细节
fm.py文件
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class FMLayer(keras.layers.Layer):
    def __init__(self, k=16, activation="sigmoid", **kwargs):
        self.lr = keras.layers.Dense(1, use_bias=True)
        self.k = k
        self.activate = keras.layers.Activation(activation)
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                        shape=(input_shape[1], self.k),
                        initializer='glorot_uniform',
                        trainable=True)

    def call(self, inp):
        # 线性模型部分（batch_size, n_features）->(batch_size, 1)
        lr = self.lr(inp)
        # (v_if * x_i)^2 (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        a = tf.reduce_sum(K.pow(K.dot(inp, self.kernel), 2), axis=1, keepdims=True)
        # (v_if^2 * x_i) (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        b = tf.reduce_sum(K.dot(inp ** 2, self.kernel ** 2), axis=1, keepdims=True)
        out = lr + a - b
        out = self.activate(out)
        return out


class FM(object):
    def __init__(self, k, n_features):
        self.fm = FMLayer(k)
        self.x_in = keras.Input(shape=(n_features,), name="inp")

    def build(self):
        self.out = self.fm(self.x_in)
        self.model = keras.Model(self.x_in, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy, metrics=["acc"], optimizer="adam")

```

### 3.3 数据处理
```python
data = pd.read_csv("./telecom-churn/vec_tel_churn.csv", header=0)
data.head()

#output:
	Unnamed: 0	customerID	gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	MultipleLines	InternetService	...	DeviceProtection	TechSupport	StreamingTV	StreamingMovies	Contract	PaperlessBilling	PaymentMethod	MonthlyCharges	TotalCharges	Churn
007590-VHVEG	0.00.01.00.01.00.02.01.0	...	0.00.00.00.00.01.00.029.8529.850.0
115575-GNVDE	1.00.00.00.034.01.00.01.0	...	1.00.00.00.01.00.01.056.951889.500.0
223668-QPYBK	1.00.00.00.02.01.00.01.0	...	0.00.00.00.00.01.01.053.85108.151.0
337795-CFOCW	1.00.00.00.045.00.02.01.0	...	1.01.00.00.01.00.02.042.301840.750.0
449237-HQITU	0.00.00.00.02.01.00.02.0	...	0.00.00.00.00.01.00.070.70151.651.0
```
数据说明如下：
前面两列都是索引，不用管。customerID表示用户编号，后面19列为用户特征，最后一列为用户流失与否的标签（0/1表示）。  
可以看到tenure、MonthlyCharges、TotalCharges这三列不是onehot编码，这里分箱后onehot处理。  

```python
tenure_cut = pd.cut(data.tenure, [0, 20, 60, 80])
data[["tenure_0","tenure_1","tenure_2"]] = pd.get_dummies(tenure_cut, prefix="tenure")
mc_cut = pd.cut(data.MonthlyCharges, [0, 40, 80, 120])
data[["mc_0", "mc_1", "mc_2"]] = pd.get_dummies(mc_cut, prefix="mc")
tc_cut = pd.cut(data.TotalCharges, [i * 1000 for i in range(10)])
data[["tc_{}".format(i) for i in range(9)]] = pd.get_dummies(tc_cut, prefix="tc")
features = data.columns[2:]
features = list(features)
print(type(features))
features.remove("tenure")
features.remove("MonthlyCharges")
features.remove("TotalCharges")
features.remove("Churn")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data["Churn"], 
                                                    test_size=.1, 
                                                    random_state=10, shuffle=True)
                                            
```

### 3.4模型训练
```python
fm_al = FM(8, 31)
fm_al.build()
dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = dataset.shuffle(len(X_train)).batch(32)

fm_al.model.fit(train_dataset, epochs=20)

#output
Train for 199 steps
Epoch 1/20
199/199 [==============================] - 1s 5ms/step - loss: 0.5924 - acc: 0.7558
Epoch 2/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4890 - acc: 0.7707
Epoch 3/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4646 - acc: 0.7777
Epoch 4/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4564 - acc: 0.7831
Epoch 5/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4409 - acc: 0.7862
Epoch 6/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4358 - acc: 0.7925
Epoch 7/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4333 - acc: 0.7938
Epoch 8/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4287 - acc: 0.7957
Epoch 9/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4265 - acc: 0.8014
Epoch 10/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4255 - acc: 0.7987
Epoch 11/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4246 - acc: 0.8004
Epoch 12/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4224 - acc: 0.7993
Epoch 13/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4234 - acc: 0.8021
Epoch 14/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4217 - acc: 0.8064
Epoch 15/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4215 - acc: 0.8025
Epoch 16/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4203 - acc: 0.8067
Epoch 17/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4183 - acc: 0.8047
Epoch 18/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4188 - acc: 0.8034
Epoch 19/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4179 - acc: 0.8051
Epoch 20/20
199/199 [==============================] - 0s 2ms/step - loss: 0.4174 - acc: 0.8051
```
可以看到训练集上最终准确率在80%左右。

### 3.5 模型评估
验证集上准确率是79%，效果不错。
```python
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
test_dataset = test_dataset.batch(32)
loss, acc = fm_al.model.evaluate(test_dataset)
#损失、准确率
loss, acc

#output
23/23 [==============================] - 0s 3ms/step - loss: 0.4003 - acc: 0.7929
(0.4002552039860545, 0.7929078)
```
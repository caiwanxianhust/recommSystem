**写在前面**：今天介绍的MLR模型是阿里提出的一个传统领域的CTR预估模型，过时挺久了，而且笔者简单试了下还翻车了，效果还不如逻辑回归，但是其中蕴含的思想值得学习下，所以写了这篇文章~

## 1 算法背景

MLR模型，全称Mixed Logistic Regression（混合逻辑回归），2012年由阿里盖坤团队提出，在当时引领了广告领域CTR预估算法的全新升级。原文[《Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction》](https://arxiv.org/pdf/1704.05194.pdf)提出了一个适用于大规模数据的分段线性模型LS-PLM（就是MLR），**核心思想是采用分而治之的策略，即先将特征空间划分为若干个局部区域，然后在每个区域拟合一个线性模型，得到加权线性预测组合的输出**。

MLR可以看做是对LR的一个自然推广，它采用分而治之的思路，用分片线性的模式来拟合高维空间的非线性分类面。简单来说，就是先计算样本属于每个分片空间的权重，在每个分片空间空间中使用逻辑回归，然后计算加权平均值作为模型预测结果，可以看作两次逻辑回归：一次多分类（分片），一次二分类（CTR预估）。

预测公式如下：
$$
p(y=1|x) = g(\sum_{j=1}^{m}\sigma (u_j^T x)\eta (w_j^T x))
$$
式中，$u$是划分分片空间的多分类权重参数，决定样本属于哪一个分片空间或者说在每一个空间的占比；$w$是分类参数，决定分片空间内的预测；超参数分片数$m$可以较好地平衡模型的拟合与推广能力。当$m=1$时MLR就退化为普通的LR，$m$越大模型的拟合能力越强，但是模型参数规模随$m$线性增长，相应所需的训练样本也随之增长。因此实际应用中$m$需要根据实际情况进行选择。原文中MLR模型用4个分片可以完美地拟合出数据中的菱形分类面。

![mlr模型示意](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5oWQScSU0of3eVP0Xl2WIyektnouRW2jxV7qgNficnUeHM1BTWqtmJe0ord6w9I1BP9vrWOLibibPmMg/0?wx_fmt=png)

上式中$\sigma$为多分类激活函数，实际应用中使用$softmax$激活函数；$\eta$为二分类激活函数，使用$sigmoid$，所以上式展开如下：
$$
p(y=1|x) = \sum_{i=1}^{m}\frac{exp(u_i^T x)}{\sum_{j=1}^{m}exp(u_j^T x)}\cdot \frac{1}{1+exp(-w_i^T x)}
$$
关于损失函数，原文采用了neg-likelihood loss function（负对数似然，其实就是交叉熵）以及L1，L2正则，形式如下：
$$
\mathop {\arg \min}\limits_{\theta}f(\theta) = loss(\theta) + \lambda ||\theta||_{2,1} + \beta ||\theta||_1
$$

## 2 模型结构和代码

相比于传统逻辑回归来说，MLR就是提出了一个分而治之的思路，加入了一个多分类模块（softmax回归），整体比较简单。下面给出笔者的实现：

mlr.py文件

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class MixedLogisticRegression(object):
    def __init__(self, n_features, num_blocks, **kwargs):
        super(MixedLogisticRegression, self).__init__(**kwargs)
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.block_layer = keras.layers.Dense(num_blocks, use_bias=False, activation="softmax",
                                              name="block_layer")
        self.lr = keras.layers.Dense(num_blocks, use_bias=False, activation="sigmoid",
                                     name="lr")
        self.multiply_layer = keras.layers.Multiply(name="multiply_layer")
        self.sum_layer = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name="sum_layer")

    def build(self):
        # [batch_size, n_features] -> [batch_size, num_blocks]
        block_out = self.block_layer(self.inp)
        # [batch_size, n_features] -> [batch_size, num_blocks]
        lr_out = self.lr(self.inp)
        # [batch_size, n_features] -> [batch_size, num_blocks]
        self.out = self.multiply_layer([block_out, lr_out])
        # [batch_size, n_features] -> [batch_size, 1]
        self.out = self.sum_layer(self.out)
        self.mlr_model = keras.Model(self.inp, self.out)
        self.mlr_model.compile(loss=keras.losses.binary_crossentropy,
                               metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()],
                               optimizer=keras.optimizers.Ftrl())
```

具体步骤如下：

- 分片层（softmax多分类）：一层全连接网络，激活函数使用softmax，其输出记为`block_out`，大小为`[batch_size, num_blocks]`，可以看作样本在每个分片空间所占比重；
- 逻辑回归：一层神经网络，激活函数使用sigmoid，其输出记为`lr_out`，大小为`[batch_size, num_blocks]`，可以看作样本在每个分片空间的分类结果；
- multiply层+sum层：将`block_out`与`lr_out`按元素相乘然后求和，其实就是将每个分片空间的预测结果加权求和计算最终的分类概率，权重是`block_out`。

**优化器**：实际应用中样本具有成千上万维度导致大量稀疏特征，通常希望模型参数更加稀疏，但是简单的L1正则无法真正做到稀疏，一些梯度截断方法（TG）的提出就是为了解决这个问题，在这其中FTRL是兼备精度和稀疏性的在线学习方法。FTRL的基本思想是将接近于0的梯度直接置零，计算时直接跳过以减少计算量。这里直接使用tf.keras中的Ftrl优化器即可。

模型结构图如下：

![mlr模型结构](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5oWQScSU0of3eVP0Xl2WIyefAibHADaNuEpnbpWUKDNcNNSfG458WOOLibPeeydkJw4Ka7R3OYia8Dicw/0?wx_fmt=png)

## 3 基于tensorflow2.X的mlr模型实践

这里使用的是Kaggle比赛——Telco Customer Churn电信客户流失数据集。

### 3.1 导入包

```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import pandas as pd
from mlr import MixedLogisticRegression

# 设置GPU显存动态增长
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
```

### 3.2 数据处理

读取数据：

```python
data = pd.read_csv("./telecom-churn/vec_tel_churn.csv", header=0)
data.head()

#output:
Unnamed: 0	customerID	gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	MultipleLines	InternetService	OnlineSecurity	OnlineBackup	DeviceProtection	TechSupport	StreamingTV	StreamingMovies	Contract	PaperlessBilling	PaymentMethod	MonthlyCharges	TotalCharges	Churn
0	0	7590-VHVEG	0.0	0.0	1.0	0.0	1.0	0.0	2.0	1.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	29.85	29.85	0.0
1	1	5575-GNVDE	1.0	0.0	0.0	0.0	34.0	1.0	0.0	1.0	1.0	0.0	1.0	0.0	0.0	0.0	1.0	0.0	1.0	56.95	1889.50	0.0
2	2	3668-QPYBK	1.0	0.0	0.0	0.0	2.0	1.0	0.0	1.0	1.0	1.0	0.0	0.0	0.0	0.0	0.0	1.0	1.0	53.85	108.15	1.0
3	3	7795-CFOCW	1.0	0.0	0.0	0.0	45.0	0.0	2.0	1.0	1.0	0.0	1.0	1.0	0.0	0.0	1.0	0.0	2.0	42.30	1840.75	0.0
4	4	9237-HQITU	0.0	0.0	0.0	0.0	2.0	1.0	0.0	2.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	70.70	151.65	1.0
```

前面两列都是索引，不用管。customerID表示用户编号，后面19列为用户特征，最后一列为用户流失与否的标签（0/1表示）。

```python
# 单值离散特征
single_discrete = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
# 多值离散特征
multi_discrete = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
# 连续数值特征
continuous = ["tenure", "MonthlyCharges", "TotalCharges"]

# 连续数值特征处理
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[continuous] = scaler.fit_transform(data[continuous])

multi_discrete_data = pd.get_dummies(data[multi_discrete], columns=multi_discrete)
data = pd.concat([data, multi_discrete_data], axis=1)
features = single_discrete + list(multi_discrete_data.columns) + continuous

# 划分训练集测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data["Churn"], 
                                                    test_size=.1, 
                                                    random_state=10, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=.1, 
                                                    random_state=10, shuffle=True)
# 洗牌、划分batch，转为可输入模型tensor
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = train_dataset.shuffle(len(X_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
val_dataset = val_dataset.shuffle(len(X_val)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
test_dataset = test_dataset.batch(32)
```

特征有40个，分类特征全部onehot处理，连续数值特征进行标准化处理。取全部数据集的10%为测试集，剩下90%为训练集和验证集（比例为9：1）

### 3.3 建立模型

分片超参数$m$定为12，这里也没有做调参处理，有兴趣可以调一下。

```python
mlral = MixedLogisticRegression(n_features=len(features), num_blocks=12)
mlral.build()
mlral.mlr_model.summary()

#output：
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
inp (InputLayer)                [(None, 40)]         0                                            
__________________________________________________________________________________________________
block_layer (Dense)             (None, 12)           480         inp[0][0]                        
__________________________________________________________________________________________________
lr (Dense)                      (None, 12)           480         inp[0][0]                        
__________________________________________________________________________________________________
multiply_layer (Multiply)       (None, 12)           0           block_layer[0][0]                
                                                                 lr[0][0]                         
__________________________________________________________________________________________________
sum_layer (Lambda)              (None, 1)            0           multiply_layer[0][0]             
==================================================================================================
Total params: 960
Trainable params: 960
Non-trainable params: 0
```

模型总共包含960个待训练参数。

```python
threshold = .5
mlral.mlr_model.compile(loss=keras.losses.binary_crossentropy,
                               metrics=[keras.metrics.BinaryAccuracy(threshold=threshold), keras.metrics.Recall(thresholds=threshold)],
                               optimizer=keras.optimizers.Ftrl())
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),]
mlral.mlr_model.fit(train_dataset, epochs=200, validation_data=val_dataset, callbacks=callbacks)

#output:
Epoch 155/200
179/179 [==============================] - 0s 2ms/step - loss: 0.4630 - binary_accuracy: 0.7821 - recall_1: 0.2701 - val_loss: 0.5119 - val_binary_accuracy: 0.7240 - val_recall_1: 0.2560
Epoch 156/200
179/179 [==============================] - 0s 2ms/step - loss: 0.4623 - binary_accuracy: 0.7828 - recall_1: 0.2735 - val_loss: 0.5111 - val_binary_accuracy: 0.7240 - val_recall_1: 0.2560
Epoch 157/200
179/179 [==============================] - 0s 3ms/step - loss: 0.4623 - binary_accuracy: 0.7831 - recall_1: 0.2754 - val_loss: 0.5116 - val_binary_accuracy: 0.7240 - val_recall_1: 0.2560
Epoch 158/200
179/179 [==============================] - 0s 3ms/step - loss: 0.4615 - binary_accuracy: 0.7830 - recall_1: 0.2761 - val_loss: 0.5105 - val_binary_accuracy: 0.7224 - val_recall_1: 0.2560
```

### 3.4 模型评估

```python
loss, acc, recall = mlral.mlr_model.evaluate(test_dataset)
loss, acc, recall

#output:
23/23 [==============================] - 0s 3ms/step - loss: 0.4362 - binary_accuracy: 0.7972 - recall_1: 0.2327
(0.4361749320574429, 0.7971631, 0.2327044)
```

效果被LR秒成渣。。。笔者试过调整超参数，效果挺差的，可能是这个数据集比较小，加上特征工程没有做所以效果很差，反正比不上其他传统模型，可能阿里有其他tricks吧。

## 4 小结

MLR算法适合于工业级的大规模稀疏数据场景问题，如广告CTR预估。背后的优势体现在两个方面：

- 端到端的非线性学习：从模型端自动挖掘数据中蕴藏的非线性模式（加入了分片思想），省去了大量的人工特征设计，这使得MLR算法可以端到端地完成训练，在不同场景中的迁移和应用非常轻松。
- 稀疏性：MLR在建模时引入了L1和L2,1范数正则，可以使得最终训练出来的模型具有较高的稀疏度， 模型的学习和在线预测性能更好。当然，这也对算法的优化求解带来了巨大的挑战。
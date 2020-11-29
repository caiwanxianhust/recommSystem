**写在前面**：本文主要参考了[用Keras实现一个DeepFM](https://blog.csdn.net/songbinxu/article/details/80151814) ，又结合了[DeepFM原文](https://arxiv.org/pdf/1703.04247.pdf)的思想，模型结构中加入不少笔者自己的思考，最终结构与前文大相径庭。供读者参考，错漏之处请多指教。

## 1 算法背景

对于一个基于CTR预估的推荐系统，学习用户点击行为背后复杂、有效的特征表示非常重要。那么，如何学习出有效的特征表示呢？

对于一阶特征，直接采用线性模型学习参数$w_i$：
$$
y = w_0 + \sum_{i=1}^{n}{w_i x_i}
$$
对于二阶特征，可以采用FM模型（或FFM模型）来学习参数$w_{ij}=\lt v_i,v_j \gt$，这里给出FM模型的二次项：
$$
y = \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{w_{ij}x_i x_j} = \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_i,v_j \gt x_i x_j}
$$
在不同的推荐场景中，低阶组合特征或者高阶组合特征可能都会对最终的CTR产生影响。理论上来讲FM可以对高阶特征组合进行建模，但实际上因为计算复杂度的原因一般都只用到了二阶特征组合。那么对于高阶的特征组合来说，我们很自然的想法，通过多层的神经网络即DNN去解决。DNN将FM模型的隐向量作为全连接层的输入，学习到embedding vector之间更为复杂的关系。也就是说Deep部分的embedding层和FM共享参数。

DeepFM就是将上述三种特征表示方式组合起来，建立三个模型：线性模型、二阶特征组合模型、全连接神经网络，然后将三个模型的输出加权求和就是最终模型的输出。

## 2 模型结构

整体的架构体系如下，左侧为FM的结构层（包含了线性模型），右侧为Deep部分的结构层，两者共用相同的特征输入。每个神经元进行了什么操作也在图中表示得很清楚。需要注意的是，**图中的连线有红线和黑线的区别**，红线表示权重为 1，黑线表示有需要训练的权重连线。  

![DeepFM模型结构](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5oJ0XLLMj8ibbGxdr2pvboTAibpt5tS9zCSkAarcB8UebTibDOs8QtNDLRgJg3j01zwvVfLad7oYdMNQ/0?wx_fmt=png)

- Addition 普通的线性加权相加，就是 w*x
- Inner Product 内积操作，就是 FM 的二次项隐向量两两相乘的部分
- Sigmoid 激活函数，即最后整合两部分输出合并进入 sigmoid 激活函数得到的输出结果
- Activation Function，这里为激活函数用的就是线性整流器 relu 函数

### 2.1 模型输入

通常样本特征主要分为以下几种情况：

1. 单值离散特征
2. 多值离散特征，如 [5,9,11]
3. 连续值特征。（如果是较大的连续值，需在特征工程部分先做归一化，或者考虑先做离散化处理成离散值）

针对以上三种情况，[用Keras实现一个DeepFM](https://blog.csdn.net/songbinxu/article/details/80151814)这篇文章给出了文章作者的处理方式，具体如下：

- **连续型field**对一次项的贡献等于自身数值乘以权值$w$，可以用`Dense(1)`层实现，任意个连续型field输入到同一个Dense层即可，因此在数据处理时，可以先将所有连续型field拼成一个大矩阵

- **其次**，**单值离散型field**根据样本特征取值的index，从$w$中取出对应权值（一阶模型为标量，二阶模型为向量），由于离散型特征值为1，故它对一次项的贡献即取出的**权值本身**。取出权值的过程称为 **table-lookup**，可以用`Embedding(n,1)`层实现（n为该field特征取值个数）。

- **最后**，**多值离散型field**可以同时取多个特征值，为了batch training，必须对样本进行补零padding。相似地可用`Embedding`层实现，Value并不是必要的，但Value可以作为mask来使用，当然也可以在`Embedding`中设置`mask_zero=True`。  

可以看到，上面的处理方式，**最大的缺点就是对每一个field都要定义对应的Input层和Embedding层**，假如有100个field，代码看起来会非常冗长，模型结构也会巨复杂，显然无法满足实际应用。因此，笔者根据自己的思考做了一些改进，听我慢慢道来。

**什么是Embedding?**

**Embedding**的实现机制其实就是两步：1.**初始化权重**$w(n\_features, embbeding\_dim)$，将这个权重设为可学习调整的类型；2. 根据特征的索引**table-lookup**出相应的向量，对应于$w[feature\_index, :]$。

举例如下，对于两个特征，特征索引分别是0，1，其Embedding过程如下：
$$
\begin{pmatrix}0 & 1 \end{pmatrix}\quad 查权重w表\begin{pmatrix}w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\\ 
w_{31} & w_{32} & w_{33}\\ 
w_{41} & w_{42} & w_{43}\\ 
w_{51} & w_{52} & w_{53}\\ 
w_{61} & w_{62} & w_{63}\end{pmatrix}=\begin{pmatrix}w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\end{pmatrix}
$$

看到了吗？一个以$2\times 6$的onehot矩阵的为输入、中间层节点数为3的全连接神经网络层，不就相当于在$w_{ij}$这个矩阵中，取出第1、2行，这不是跟上述所谓的Embedding的查表（从表中找出对应特征的向量）是一样的吗？事实上，正是如此！这就是所谓的Embedding层，Embedding层就是以onehot为输入、中间层节点为特征向量维数的全连接层！而这个全连接层的参数，就是一个“Embedding权重表”！

$$
\begin{pmatrix}1 & 0 & 0 & 0 & 0 & 0\\ 
0 & 1 & 0 & 0 & 0 & 0 \end{pmatrix}\begin{pmatrix}w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\\ 
w_{31} & w_{32} & w_{33}\\ 
w_{41} & w_{42} & w_{43}\\ 
w_{51} & w_{52} & w_{53}\\ 
w_{61} & w_{62} & w_{63}\end{pmatrix}=\begin{pmatrix}w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\end{pmatrix}
$$

由此可见，对于离散特征而言，可以通过onehot+Dense的方式实现Embedding的效果。而且对于输入数据而言，onehot之后数据对齐，避免了使用index输入时多值离散特征的padding操作，因此可以把所有特征放在一个Input层即可。两种方法参数规模是一样的，每个离散特征的值都有一个相应的待训练权重与之对应（单值离散特征虽然有两个值0/1，但只有一个权重）。本文将以电信用户流失数据集作为案例进行说明。

#### 2.1.1 读取数据

```python
data = pd.read_csv("./telecom-churn/vec_tel_churn.csv", header=0)
data.head()

#output:

Unnamed: 0	customerID	gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	MultipleLines	InternetService	...	DeviceProtection	TechSupport	StreamingTV	StreamingMovies	Contract	PaperlessBilling	PaymentMethod	MonthlyCharges	TotalCharges	Churn
0	0	7590-VHVEG	0.0	0.0	1.0	0.0	1.0	0.0	2.0	1.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	29.85	29.85	0.0
1	1	5575-GNVDE	1.0	0.0	0.0	0.0	34.0	1.0	0.0	1.0	...	1.0	0.0	0.0	0.0	1.0	0.0	1.0	56.95	1889.50	0.0
2	2	3668-QPYBK	1.0	0.0	0.0	0.0	2.0	1.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	1.0	1.0	53.85	108.15	1.0
3	3	7795-CFOCW	1.0	0.0	0.0	0.0	45.0	0.0	2.0	1.0	...	1.0	1.0	0.0	0.0	1.0	0.0	2.0	42.30	1840.75	0.0
4	4	9237-HQITU	0.0	0.0	0.0	0.0	2.0	1.0	0.0	2.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	70.70	151.65	1.0
```

前面两列都是索引，不用管。customerID表示用户编号，后面19列为用户特征，最后一列为用户流失与否的标签（0/1表示）。

```python
data.columns

#output:
Index(['Unnamed: 0', 'customerID', 'gender', 'SeniorCitizen', 'Partner',
       'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
       'Churn'],
      dtype='object')
```

主要有以下三种特征：单值离散特征、多值离散特征、连续数值特征

```python
# 单值离散特征
single_discrete = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
# 多值离散特征
multi_discrete = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
# 连续数值特征
continuous = ["tenure", "MonthlyCharges", "TotalCharges"]
len(single_discrete), len(multi_discrete), len(continuous)

#output:
(6, 10, 3)
```

#### 2.1.2 连续数值型特征处理

对于连续数值型特征，这里做标准化处理。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[continuous] = scaler.fit_transform(data[continuous]
data[continuous].head()

#output:
	tenure	MonthlyCharges	TotalCharges
0	-1.277445	-1.160323	-0.994971
1	0.066327	-0.259629	-0.173876
2	-1.236724	-0.362660	-0.960399
3	0.514251	-0.746535	-0.195400
4	-1.236724	0.197365	-0.941193
```

可以看到，标准化处理后，tenure、MonthlyCharges、TotalCharges三列已经呈$N(0,1)$的正态分布。

#### 2.1.3 多值离散特征处理

对于多值离散特征，做onehot处理，这里使用pd.get_dummies()函数。

```python
multi_discrete_data = pd.get_dummies(data[multi_discrete], columns=multi_discrete)
multi_discrete_data.head()

#output:
	MultipleLines_0.0	MultipleLines_1.0	MultipleLines_2.0	InternetService_0.0	InternetService_1.0	InternetService_2.0	OnlineSecurity_0.0	OnlineSecurity_1.0	OnlineSecurity_2.0	OnlineBackup_0.0	...	StreamingMovies_0.0	StreamingMovies_1.0	StreamingMovies_2.0	Contract_0.0	Contract_1.0	Contract_2.0	PaymentMethod_0.0	PaymentMethod_1.0	PaymentMethod_2.0	PaymentMethod_3.0
0	0	0	1	0	1	0	1	0	0	0	...	1	0	0	1	0	0	1	0	0	0
1	1	0	0	0	1	0	0	1	0	1	...	1	0	0	0	1	0	0	1	0	0
2	1	0	0	0	1	0	0	1	0	0	...	1	0	0	1	0	0	0	1	0	0
3	0	0	1	0	1	0	0	1	0	1	...	1	0	0	0	1	0	0	0	1	0
4	1	0	0	0	0	1	1	0	0	1	...	1	0	0	1	0	0	1	0	0	0
5 rows × 31 columns
```

可以看到原本10维的特征膨胀为31维，每列值只取0/1。

#### 2.1.4 特征拼接

```python
# 将onehot后的多值离散特征拼接到原数据
data = pd.concat([data, multi_discrete_data], axis=1)
# 特征列表=单值离散特征 + 多值离散特征 + 连续数值特征
features = single_discrete + list(multi_discrete_data.columns) + continuous
len(features)

#output：
40
```

### 2.2 线性模型

$$
w_0 + \sum_{i=1}^{n}{w_i x_i}
$$

一层神经元数量为1的Dense即可。核心代码如下：  

```python
self.inp = keras.Input(shape=(n_features,), name="inp")
self.lr = keras.layers.Dense(1, use_bias=True, name="lr")
self.lr_out = self.lr(self.inp)
```

### 2.3 二阶特征交互模型

$$
\begin{split}
\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_i, v_j \gt x_i x_j} &= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\lt v_i, v_j \gt x_i x_j}- \frac{1}{2}\sum_{i=1}^{n}{\lt v_i, v_i \gt x_i x_i}\\ 
&= \frac{1}{2}(\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}{v_{i,f} v_{j,f} x_i x_j} - \sum_{i=1}^{n}\sum_{f=1}^{k}{v_{i,f}^2 x_i^2})\\ 
&= \frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f} x_i)(\sum_{j=1}^{n}v_{j,f} x_j) - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2})\\ 
&= \frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f} x_i)^2 - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2})
\end{split}
$$

这里比较复杂，主要分为两部分：先求和再平方、先平方再求和。

**前文方法**：

假设$V$矩阵大小是 `[max_feat, K]`，$X$矩阵大小是 `[batch_size, max_len]`，则先求Embedding $VX$，大小为 `[batch_size, F, K]`（这里的`F`是所有field拼接后的最长长度）。求和项内部第一项$(\sum_{i=1}^{n}v_{i,f} x_i)^2$，即Embedding先在第1维求和变成`[batch_size, K]`，然后逐元素求平方（还是`[batch_size, K]`）；第二项 $\sum_{i=1}^{n}{v_{i,f}^2 x_i^2}$ 是Embedding先逐元素求平方（还是`[batch_size, F, K]`），再对第一维求和，变成`[batch_size, K]`。两项相减之后除以2，对第1维求和，变成`[batch_size, 1]`，即各样本二次项的值。

这里前文由于使用了Embedding的方式，所以在处理第二项时较为简洁，缺点只是Input层较为庞大。

**笔者思考**：

（1）$(\sum_{i=1}^{n}v_{i,f} x_i)^2$：很显然就是一个Dense层，然后把结果逐元素平方；

（2）$\sum_{i=1}^{n}{v_{i,f}^2 x_i^2}$：也是一个Dense层，输入是$x^2$，权重是$v^2$。先进行$x$平方很容易做到，如何对公共权重平方后作为新的Dense层权重呢？这里我们采用自定义层解决这个问题，事实上在笔者以前的文章[【推荐算法】FM模型原理和实践](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247483745&idx=2&sn=cf64c279abab05a2009e038f1c774517&chksm=e92784d8de500dce3e68c9efe6078af6ff8716695f290a38b80097fdaa126919afcbd3f469b3&token=1195106363&lang=zh_CN#rd)中已经实现过了。代码如下：

```python
class InteractionLayer(keras.layers.Layer):
    def __init__(self, factor_dim=16, **kwargs):
        self.factor_dim = factor_dim
        super(InteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inp):
        # (v_if * x_i)^2 (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        a = tf.reduce_sum(K.pow(K.dot(inp, self.kernel), 2), axis=1, keepdims=True)
        # (v_if^2 * x_i) (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        b = tf.reduce_sum(K.dot(inp ** 2, self.kernel ** 2), axis=1, keepdims=True)
        interaction_out = a - b
        # (batch_size, n_features） -> (batch_size, n_features * k）
        rep_inp = K.repeat_elements(inp, rep=self.factor_dim, axis=1)
        # (n_features, k) -> (1, n_features * k)
        flatten_kernel = tf.reshape(self.kernel, shape=(1, -1))
        # (batch_size, n_features * k)
        flatten_inp_emb = rep_inp * flatten_kernel
        return (interaction_out, flatten_inp_emb)
```

这里交互层输出了两项内容，第一项就是要计算的二阶特征交互模型的结果，第二项内容将作为DNN部分的输入，稍后介绍。

### 2.3 全连接神经网络（DNN）

前面讲过，DNN将FM模型的隐向量Flatten之后作为全连接层的输入，每一层Dense后面可以加上dropout防止过拟合。

前文使用了Embedding方法，在将所有Embedding拼接之后，可以很轻松地直接执行flatten操作，将`[batch_size, n_features, facto_dim]`转换为`[batch_size, n_features*facto_dim]`，而笔者没有使用Embedding改如何获得DNN的输入呢？

这里笔者使用了repeat_elements()函数，先将输入`[batch_size, n_features]`的第1维复制factor_dim倍，形状变为`[batch_size, n_features*facto_dim]`，再将交互层的公共权重由`[n_features，facto_dim]`reshape为`[1, n_features*facto_dim]`，最后将两者相乘，也得到了同样的效果。具体代码如下：

```python
# (batch_size, n_features） -> (batch_size, n_features * k）
rep_inp = K.repeat_elements(inp, rep=self.factor_dim, axis=1)
# (n_features, k) -> (1, n_features * k)
flatten_kernel = tf.reshape(self.kernel, shape=(1, -1))
# (batch_size, n_features * k)
flatten_inp_emb = rep_inp * flatten_kernel
```

输入有了，然后就是朴实无华的多层Dense了，由于数据集比较小，笔者减少了Dense层得到神经元数量。

```python
self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.flatten_inp_emb))
self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1, activation='relu')(self.deep_out))
```

### 2.4 模型输出

最后将以上三个模型的结果拼接起来加一层逻辑回归就是最终的模型输出：

```python
self.out = self.concate_out_layer([self.lr_out, self.interaction_out, self.deep_out])
self.out = self.com_dense(self.out)
self.model = keras.Model(self.inp, self.out)
self.model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["acc"])
```

### 2.5 模型整体结构和代码

deepfm.py

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class InteractionLayer(keras.layers.Layer):
    def __init__(self, factor_dim=16, **kwargs):
        self.factor_dim = factor_dim
        super(InteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inp):
        # (v_if * x_i)^2 (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        a = tf.reduce_sum(K.pow(K.dot(inp, self.kernel), 2), axis=1, keepdims=True)
        # (v_if^2 * x_i) (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        b = tf.reduce_sum(K.dot(inp ** 2, self.kernel ** 2), axis=1, keepdims=True)
        interaction_out = a - b
        # (batch_size, n_features） -> (batch_size, n_features * k）
        rep_inp = K.repeat_elements(inp, rep=self.factor_dim, axis=1)
        # (n_features, k) -> (1, n_features * k)
        flatten_kernel = tf.reshape(self.kernel, shape=(1, -1))
        # (batch_size, n_features * k)
        flatten_inp_emb = rep_inp * flatten_kernel
        return (interaction_out, flatten_inp_emb)


class DeepFM:
    def __init__(self, n_features, factor_dim):
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.lr = keras.layers.Dense(1, use_bias=True, name="lr")
        self.interaction = InteractionLayer(factor_dim, name="interaction")
        self.concate_inp_layer = keras.layers.Concatenate(axis=-1, name="concate_inp_layer")
        self.concate_out_layer = keras.layers.Concatenate(axis=-1, name="concate_out_layer")
        self.com_dense = keras.layers.Dense(1, activation="sigmoid")

    def build(self):
        # (batch_size, n_features) -> (batch_size, 1)
        self.lr_out = self.lr(self.inp)
        # (batch_size, n_features) -> (batch_size, 1)
        self.interaction_out, self.flatten_inp_emb = self.interaction(self.inp)
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.flatten_inp_emb))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1, activation='relu')(self.deep_out))
        self.out = self.concate_out_layer([self.lr_out, self.interaction_out, self.deep_out])
        self.out = self.com_dense(self.out)
        self.model = keras.Model(self.inp, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["acc"])

```

结构图如下：

```
keras.utils.plot_model(deepfm.model, "deepfm.png", show_layer_names=True, show_shapes=True)
```



![deepfm](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qSqAbfHoK5TLJabZywAAw7dzo0pkCFEpagaD3P0ExkVn3WpjgeIyJWZRfLNCo20ibLh3w2rX0VZsQ/0?wx_fmt=png)

可见，经过笔者改造后模型结构可读性提高了不少。

## 3 案例实践：基于DeepFM预估电信用户流失

### 3.1 导入库

```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from deepfm import DeepFM
import numpy as np
import pandas as pd

# 设置GPU显存动态增长
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
```

## 3.2 输入数据处理

```python
data = pd.read_csv("./telecom-churn/vec_tel_churn.csv", header=0)
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

# 多值离散特征处理
multi_discrete_data = pd.get_dummies(data[multi_discrete], columns=multi_discrete)
data = pd.concat([data, multi_discrete_data], axis=1)
features = single_discrete + list(multi_discrete_data.columns) + continuous

# 划分训练集测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data["Churn"], 
                                                    test_size=.1, 
                                                    random_state=10, shuffle=True)

# 洗牌、划分batch，转为可输入模型tensor
dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = dataset.shuffle(len(X_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
test_dataset = test_dataset.batch(32)
```

### 3.3 建立模型

```python
factor_dim = 8
deepfm = DeepFM(len(features), factor_dim)
deepfm.build()
deepfm.model.summary()

#output:
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
inp (InputLayer)                [(None, 40)]         0                                            
__________________________________________________________________________________________________
interaction (InteractionLayer)  ((None, 1), (None, 3 320         inp[0][0]                        
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8)            2568        interaction[0][1]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 8)            0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 8)            72          dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 8)            0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 8)            72          dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 8)            0           dense_3[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            9           dropout_2[0][0]                  
__________________________________________________________________________________________________
lr (Dense)                      (None, 1)            41          inp[0][0]                        
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 1)            0           dense_4[0][0]                    
__________________________________________________________________________________________________
concate_out_layer (Concatenate) (None, 3)            0           lr[0][0]                         
                                                                 interaction[0][0]                
                                                                 dropout_3[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            4           concate_out_layer[0][0]          
==================================================================================================
Total params: 3,086
Trainabe params: 3,086
Non-trainable params: 0
```

模型总共包含3086个待训练参数，训练集大小才6000，有点少。

```
deepfm.model.fit(train_dataset, epochs=20)

#output:
Train for 199 steps
Epoch 1/20
199/199 [==============================] - 1s 3ms/step - loss: 0.5523 - acc: 0.7419
Epoch 2/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4528 - acc: 0.7857
Epoch 3/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4314 - acc: 0.7949
Epoch 4/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4250 - acc: 0.8004
Epoch 5/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4202 - acc: 0.8014
Epoch 6/20
199/199 [==============================] - 0s 997us/step - loss: 0.4183 - acc: 0.8040
Epoch 7/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4168 - acc: 0.8048
Epoch 8/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4156 - acc: 0.8034
Epoch 9/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4185 - acc: 0.8072
Epoch 10/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4138 - acc: 0.8036
Epoch 11/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4164 - acc: 0.8051
Epoch 12/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4144 - acc: 0.8058
Epoch 13/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4109 - acc: 0.8067
Epoch 14/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4124 - acc: 0.8078
Epoch 15/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4122 - acc: 0.8069
Epoch 16/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4111 - acc: 0.8083
Epoch 17/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4106 - acc: 0.8072
Epoch 18/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4109 - acc: 0.8078
Epoch 19/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4125 - acc: 0.8104
Epoch 20/20
199/199 [==============================] - 0s 1ms/step - loss: 0.4103 - acc: 0.8080
```

最终准确率在81%左右，和FFM差不多，实际工程中经过大量数据集训练及复杂的调参工作后，性能应该会有大幅度提升。

### 3.4 模型评估

```python
loss, acc = deepfm.model.evaluate(test_dataset)
loss, acc

#output:
23/23 [==============================] - 0s 5ms/step - loss: 0.3783 - acc: 0.8028
(0.37831568252295256, 0.8028369)
```

验证集上准确率为80.3%，和训练集差不多。
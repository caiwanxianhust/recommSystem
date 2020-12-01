**写在前面**：笔者前段时间一直在外地出差，已经半个多月没有更新文章了！！！打工人想保持稳定的学习效率太难了。

## 1 什么是AttentionFM？

AttentionFM（Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks，简称AFM），2017年由浙大与新加坡国立大学合作推出，顾名思义就是在FM模型中加入了Attention机制。

### 1.1 FM模型回顾

$$
FM模型：\quad y = w_0 + \sum_{i=1}^{n} {w_i x_i} + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_i, v_j \gt x_i x_j}
$$

FM全称Factorization Machine，给每个特征学习一个隐向量，通过隐向量内积来对每一组交叉特征进行建模。

从FM原理可以看出，FM存在下面两个问题：

- 特征组合时每个特征针对其他不同特征都使用同一个隐向量。所以后来提出FFM模型，将特征划分为若干个field，每个field的特征和其他field的特征组合时单独采用一个隐向量，很好地解决了隐向量单一的问题。
- 所有组合特征的权重w都有着相同的权重，即所有特征组合的重要性相同。AFM就是用来解决这个问题的。

在一次预测中，并不是所有的特征组合都是有用的。AFM模型引入attention机制，针对不同的特征组合使用不同的权重，增强了模型的表达能力。这也使得模型的可解释性更强，方便后续针对重要的特征组合进行深入研究。

### 1.2 AFM模型

在FM、DeepFM、NFM等模型中，不同field的特征Embedding向量经过特征交叉后，将各交叉特征向量按照Embedding Size维度进行“加和”，相当于是“平等”地对待所有交叉特征，未考虑特征对结果的影响程度，事实上消除了大量有价值的信息。

针对以上情况，**attention机制**被引入推荐系统中，attention机制相当于一个加权平均，attention的值就是其中权重，用来描述不同特征之间交互的重要性。例如：如果应用场景为“预测一位男性用户是否购买鼠标的可能性”，那么“性别=男”和“购买历史包含键盘”这个交叉特征比“性别=男”和“用户年龄=25”这一交叉特征更重要，模型在前一交叉特征上投入了更多的“注意力”。

这里给出AFM模型的预测公式：
$$
y_{AFM} = w_0 + \sum_{i=1}^{n}{w_i x_i} + p^T \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{a_{ij} (v_i \odot v_j)  x_i x_j}
$$

式中，$v_i，v_j$表示特征$i,j$对应field的隐向量；$\odot$的含义是element-wise product，即两个Embedding特征之间的元素积操作：
$$
(x_{1,1},x_{1,2},...) \odot (x_{2,1},x_{2,2},...) = (x_{1,1} x_{2,1},x_{1,2} x_{2,2},...)
$$
$a_{ij}$是attention的值，即注意力得分。这里使用单层的全连接网络进行参数学习。
$$
a_{ij}^{\prime} = h^T ReLU(W(v_i \odot v_j) x_i x_j + b)
$$

$$
a_{ij} = \frac {exp(a_{ij}^{\prime})}{\sum_{st} exp(a_{st}^{\prime})}
$$

其中$W\in R^{t \times f}, b \in R^{t\times 1}, h\in R^{t\times 1}$，$f$为embedding后的向量的维度，$t$为attention network的隐藏维度。

从宏观来看，AFM只是在FM的基础上添加了attention的机制，但是实际上，由于最后的加权累加，二次项并没有进行更深的网络去学习非线性交叉特征，所以它的上限和FFM很接近，没有完全发挥出DNN的优势。

## 2 模型结构

AFM模型是在“特征交叉层”和“输出层”之间引入注意力网络实现的，注意力网络的作用为每一个交叉特征提供权重。AFM模型结构如下图所示。

![AFM模型结构](https://mmbiz.qpic.cn/mmbiz_jpg/GJUG0H1sS5rNgXft60TkTjCic2j0E1WiaHDm6zwXhbCVibaib0ezXvxiaWTZPTHACthAlgmwSpImpLEPordhMCibJIXA/0?wx_fmt=jpeg)

值得注意的是，上图忽略了FM中的lr部分，只是针对FM的二次项部分进行了结构的展示。图中的前三部分：**sparse input**（输入层），**embedding layer**（嵌入层），**pair-wise interaction layer**（交互层），都和FM是一样的。而后面的两部分，则是AFM的创新所在。从比较宏观的角度理解，AFM就是通过一个attention net生成一个关于特征交叉项的权重，然后将FM原来的二次项直接累加，变成加权累加。

### 2.1 sparse input（输入层）和embedding layer（嵌入层）

通常样本特征field主要分为以下几种情况：

1. 二元离散特征，如用户性别、是否购买某个产品，输入值为0或1
2. 多值离散特征，如用户年龄段（小孩、年轻人、老年人），输入值为0或1或2
3. 连续值特征，直接输入。（如果是较大的连续值，需在特征工程部分先做归一化，或者考虑先做离散化处理成离散值）

笔者前面在其他文章中说过，Embedding其实就是onehot+Dense。因此，简单起见这里笔者自定义了一个embedding层，实现特征嵌入。

```python
class CateEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, **kwargs):
        self.emb_dim = emb_dim
        super(CateEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        :param input_shape: field个数
        :return:
        """
        self.kernel = self.add_weight(name='cate_em_vecs',
                                      shape=(input_shape[1], self.emb_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        x = K.expand_dims(x, axis=2)
        x = K.repeat_elements(x, rep=self.emb_dim, axis=2)
        out = x * self.kernel
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.emb_dim)
```

具体步骤如下：

- 先定义一个kernel，即embedding矩阵，大小为`[n_features, emb_dim]`；
- 将输入层扩展一个维度由`[batch_size, n_features]`变为`[batch_size, n_features, 1]`，然后对将其第2列复制emb_dim倍，大小变为`[batch_size, n_features, emb_dim]`；
- 让上述两个矩阵按元素相乘得到embedding层的输出。

### 2.2 Pair-wise Interaction Layer（交互层）

这一层主要是对组合特征进行建模，原来的$m$个嵌入向量，通过element-wise product（哈达玛积）操作得到了$m(m-1)/2$个组合向量，这些向量的维度和嵌入向量的维度相同均为emb_dim。形式化如下：
$$
(v_i \odot v_j)  x_i x_j
$$
也就是说Pair-wise Interaction Layer的输入是所有嵌入向量，输出也是一组向量。输出是任意两个嵌入向量的element-wise product。任意两个嵌入向量都组合得到一个Interacted vector，所以$m$个嵌入向量得到$m(m-1)/2$个向量。

如果不考虑Attention机制，在Pair-wise Interaction Layer之后直接得到最终输出，可以形式化如下：
$$
\hat y = p^T \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{ (v_i \odot v_j)  x_i x_j}
$$
其中$p$是权重矩阵。当$p$全为1的时候，很明显，这特么就是FM。

这里笔者没有像其他文章中一样使用两层循环实现Pair-wise Interaction Layer，原因很简单，两层循环计算起来又慢又不优雅，笔者是个讲究人。这里给出笔者的实现：

```python
class PairWiseInteraction(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairWiseInteraction, self).__init__(**kwargs)

    def call(self, x):
        """
        pair-wise interaction layer
        :param x: (batch_size, n_features, emb_dim)
        :return:
        """
        # (batch_size, n_features, emb_dim) -> (batch_size, n_features, 1, emb_dim)
        x = K.expand_dims(tf.cast(x, tf.float32), axis=2)
        # (batch_size, n_features, 1, emb_dim) -> (batch_size, n_features, n_features, emb_dim)
        x = K.repeat_elements(x, rep=x.shape[1], axis=2)
        xt = tf.transpose(x, perm=[0, 2, 1, 3])
        out = x * xt
        # (1, emb_dim, n_features, n_features)
        mask = 1 - tf.linalg.band_part(tf.ones((1, out.shape[3], out.shape[1], out.shape[2])), -1, 0)
        # (1, n_features, n_features, emb_dim)
        mask = tf.transpose(mask, perm=[0, 2, 3, 1])
        out = out * mask
        return tf.reshape(out, shape=(-1, out.shape[1] * out.shape[2], out.shape[3]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[1], input_shape[2])

```

笔者将输入扩充后转置，然后和转置前的输入按元素相乘，相当于用矩阵运算方法实现了Pair-wise Interaction Layer，但是相乘后的输出是一个对称矩阵，包含了$v_i \odot v_j$、$v_i \odot v_i$和$v_j \odot v_i$这三种情况，因此有$m^2$个向量。实际应用中我们只需要使用其上三角矩阵或下三角矩阵即可。因此笔者定义了一个mask矩阵，将重复的交互向量mask掉，不影响后面softmax即可。

详细步骤如下：

- 将输入扩展一个维度由`[batch_size, n_features, emb_dim]`变为`[batch_size, n_features, 1, emb_dim]`，然后对将其第2列复制n_features倍，大小变为`[batch_size, n_features, n_features, emb_dim]`，记为`x`；
- 将上一步扩充后的结果`x`进行转置，记为`xt`，将`x`与`xt`按元素相乘，得到大小为`[batch_size, n_features, n_features, emb_dim]`的矩阵，记为`out`，这个矩阵的含义就是每两个特征的的交互向量，包含$v_i \odot v_j$、$v_i \odot v_i$和$v_j \odot v_i$这三种情况；
- 定义一个大小为`[1, emb_dim, n_features, n_features]`的上三角矩阵，对矩阵进行转置使之尺寸与`out`的尺寸对应，记为`mask`；
- 将`out`与`mask`按元素相乘，然后reshape至`[batch_size, n_features * n_features, emb_dim]`，得到$m^2$个交互向量，经过mask后其中非零向量只有$m(m-1)/2$个。

### 2.3 Attention-based Pooling注意力层

在FM中，特征向量进行两两交叉之后，直接进行sum pooling，将二阶交叉向量进行等权重求和处理。但是直觉上来说，不同的交叉特征应该有着不同的重要性。不重要的交叉特征应该降低其权重，而重要性高的交叉特征应该提高其权重。Attention概念与该思想不谋而合，AFM作者顺势将其引入到模型之中，为每个交叉特征引入重要性权重，最终在对特征向量进行sum pooling时，利用重要性权重对二阶交叉特征进行加权求和。

#### 2.3.1 Attention score

为了计算特征重要性权重Attention score，作者构建了一个Attention Network，其本质是含有一个隐藏层的多层感知机（MLP）。表达式如下：
$$
a_{ij}^{\prime} = h^T ReLU(W(v_i \odot v_j) x_i x_j + b)
$$

$$
a_{ij} = \frac {exp(a_{ij}^{\prime})}{\sum_{st} exp(a_{st}^{\prime})}
$$

其中$W\in R^{t \times f}, b \in R^{t\times 1}, h\in R^{t\times 1}$，$f$为embedding后的向量的维度，$t$为attention network的隐藏维度。计算得到的$a_{ij}$即表示对应的二阶交叉特征$(v_i \odot v_j)  x_i x_j$的重要性权重Attention score。

可以看到，本文中的Attention network实际上就是一个一层神经网络：

- 输入层是嵌入向量element-wise product之后的结果(interacted vector，交互层的输出，用来在嵌入空间中对组合特征进行编码)；
- 隐藏层神经元的个数为$f$，权重和偏置项分别为$W,b$，激活函数使用ReLU；
- 输出层参数为$h^T$，神经元个数是1，不加偏置项，得到未归一化的Attention score，即$a_{ij}^{\prime}$；
- 最后，使用softmax对$a_{ij}^{\prime}$进行规范化，得到$a_{ij}$。

这里直接给出笔者的实现，不再做详细介绍。

```python
self.attention_wb = keras.layers.Dense(attention_dim, use_bias=True, activation="relu", name="attention_wb")
self.attention_h = keras.layers.Dense(1, use_bias=False, name="attention_h")
self.attention_softmax = keras.layers.Lambda(lambda x: K.softmax(x, axis=1), name="attention_softmax")

# (batch_size, n_features * n_features, attention_size)
a_out = self.attention_wb(wise_product)
# (batch_size, n_features * n_features, 1)
a_out = self.attention_h(a_out)
# (batch_size, n_features * n_features, 1)
a_out = self.attention_softmax(a_out)
```

#### 2.3.2 sum pooling

这一步对二阶交叉特征进行加权sum pooling，可以表示为$p^T \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{a_{ij} (v_i \odot v_j)  x_i x_j}$。

主要分为两步：

- 加权求和。直接将attention score与interacted vector按元素相乘，然后在第一列求和，得到大小为`[batch_size, emb_dim]`的向量；
- 池化。使用一层Dense将向量由`[batch_size, emb_dim]`pooling成`[batch_size, 1]`。

这里直接给出笔者的实现，不再做详细介绍。

```python
class AttentionPSum(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.p_sum_layer = keras.layers.Dense(1, use_bias=False)
        super(AttentionPSum, self).__init__(**kwargs)

    def call(self, inp):
        """
        :param inp: attention_out:(batch_size, n_features * n_features, 1);
                    wise_product:(batch_size, n_features * n_features, emb_dim)
        :return:
        """
        attention_out, wise_product = inp
        # (batch_size, emb_dim)
        out = tf.reduce_sum(tf.multiply(attention_out, wise_product), axis=1)
        # (batch_size, 1)
        out = self.p_sum_layer(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], 1)

```

### 2.4 输出层

这一层没什么好说的，就是把前面一阶项、偏置项以及Attention层的输出加起来，然后softmax激活即可。

### 2.5 模型整体结构和代码

基于tensorflow2.X实现，直接使用keras也能跑通。

afm.py

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class CateEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, **kwargs):
        self.emb_dim = emb_dim
        super(CateEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        :param input_shape: field个数
        :return:
        """
        self.kernel = self.add_weight(name='cate_em_vecs',
                                      shape=(input_shape[1], self.emb_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        x = K.expand_dims(x, axis=2)
        x = K.repeat_elements(x, rep=self.emb_dim, axis=2)
        out = x * self.kernel
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.emb_dim)


class PairWiseInteraction(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairWiseInteraction, self).__init__(**kwargs)

    def call(self, x):
        """
        pair-wise interaction layer
        :param x: (batch_size, n_features, emb_dim)
        :return:
        """
        # (batch_size, n_features, emb_dim) -> (batch_size, n_features, 1, emb_dim)
        x = K.expand_dims(tf.cast(x, tf.float32), axis=2)
        # (batch_size, n_features, 1, emb_dim) -> (batch_size, n_features, n_features, emb_dim)
        x = K.repeat_elements(x, rep=x.shape[1], axis=2)
        xt = tf.transpose(x, perm=[0, 2, 1, 3])
        out = x * xt
        # (1, emb_dim, n_features, n_features)
        mask = 1 - tf.linalg.band_part(tf.ones((1, out.shape[3], out.shape[1], out.shape[2])), -1, 0)
        # (1, n_features, n_features, emb_dim)
        mask = tf.transpose(mask, perm=[0, 2, 3, 1])
        out = out * mask
        return tf.reshape(out, shape=(-1, out.shape[1] * out.shape[2], out.shape[3]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[1], input_shape[2])


class AttentionPSum(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.p_sum_layer = keras.layers.Dense(1, use_bias=False)
        super(AttentionPSum, self).__init__(**kwargs)

    def call(self, inp):
        """

        :param inp: attention_out:(batch_size, n_features * n_features, 1);
                    wise_product:(batch_size, n_features * n_features, emb_dim)
        :return:
        """
        attention_out, wise_product = inp
        # (batch_size, emb_dim)
        out = tf.reduce_sum(tf.multiply(attention_out, wise_product), axis=1)
        # (batch_size, 1)
        out = self.p_sum_layer(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], 1)


class AttentionFM(object):
    def __init__(self, n_features, emb_dim=16, attention_dim=16):
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.embedding_layer = CateEmbedding(emb_dim, name="embedding_layer")
        self.lr_layer = keras.layers.Dense(1, use_bias=True, name="lr")
        self.wise_product_layer = PairWiseInteraction(name="wise_product_interaction")
        self.attention_wb = keras.layers.Dense(attention_dim, use_bias=True, activation="relu", name="attention_wb")
        self.attention_h = keras.layers.Dense(1, use_bias=False, name="attention_h")
        self.attention_softmax = keras.layers.Lambda(lambda x: K.softmax(x, axis=1), name="attention_softmax")
        self.attention_psum = AttentionPSum(name="psum")
        self.add_lr_ap = keras.layers.Add()
        self.sigmoid = keras.layers.Activation("sigmoid")

    def build(self):
        # (batch_size, n_features) -> (batch_size, 1)
        lr = self.lr_layer(self.inp)
        # (batch_size, n_features) -> (batch_size, n_features, emb_dim)
        x = self.embedding_layer(self.inp)
        # (batch_size, n_features * n_features, emb_dim)
        wise_product = self.wise_product_layer(x)
        # (batch_size, n_features * n_features, attention_size)
        a_out = self.attention_wb(wise_product)
        # (batch_size, n_features * n_features, 1)
        a_out = self.attention_h(a_out)
        # (batch_size, n_features * n_features, 1)
        a_out = self.attention_softmax(a_out)
        # (batch_size, n_features * n_features, emb_dim) -> (batch_size, emb_dim) ->(batch_size, 1)
        p_sum = self.attention_psum([a_out, wise_product])
        self.out = self.add_lr_ap([lr, p_sum])
        self.out = self.sigmoid(self.out)
        self.model = keras.Model(self.inp, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer="adam",
                           metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()])

```

模型结构图如下：

```python
keras.utils.plot_model(afmal.model, "afm.png", show_layer_names=True, show_shapes=True)
```

![afm模型结构](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rgXBrDn12Q95sFoYVPMHicDT53UmtyC0YNm7IlELTQTyHfosBCMFeAicap5n2ydibr7OXnfAkkJ6WNQ/0?wx_fmt=png)

## 3 案例实践：基于AFM预估电信用户流失

### 3.1 导入库

```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import pandas as pd
from afm import AttentionFM

# 设置GPU显存动态增长
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
```

### 3.2 输入数据处理
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

# 连续数值特征标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[continuous] = scaler.fit_transform(data[continuous])

features = single_discrete + multi_discrete + continuous

# 划分训练集测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data["Churn"], 
                                                    test_size=.1, 
                                                    random_state=10, shuffle=True)

# 洗牌、划分batch，转为可输入模型tensor
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = train_dataset.shuffle(len(X_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
test_dataset = test_dataset.batch(32)
```

### 3.3 建立模型

```python
n_features, emb_dim, attention_dim = len(features), 16, 16
afmal =AttentionFM(n_features, emb_dim, attention_dim)
afmal.build()
afmal.model.summary()

#output:
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
inp (InputLayer)                [(None, 19)]         0                                            
__________________________________________________________________________________________________
embedding_layer (CateEmbedding) (None, 19, 16)       304         inp[0][0]                        
__________________________________________________________________________________________________
wise_product_interaction (PairW (None, 361, 16)      0           embedding_layer[0][0]            
__________________________________________________________________________________________________
attention_wb (Dense)            (None, 361, 16)      272         wise_product_interaction[0][0]   
__________________________________________________________________________________________________
attention_h (Dense)             (None, 361, 1)       16          attention_wb[0][0]               
__________________________________________________________________________________________________
attention_softmax (Lambda)      (None, 361, 1)       0           attention_h[0][0]                
__________________________________________________________________________________________________
lr (Dense)                      (None, 1)            20          inp[0][0]                        
__________________________________________________________________________________________________
psum (AttentionPSum)            (None, 1)            16          attention_softmax[0][0]          
                                                                 wise_product_interaction[0][0]   
__________________________________________________________________________________________________
add (Add)                       (None, 1)            0           lr[0][0]                         
                                                                 psum[0][0]                       
__________________________________________________________________________________________________
activation (Activation)         (None, 1)            0           add[0][0]                        
==================================================================================================
Total params: 628
Trainable params: 628
Non-trainable params: 0
```

模型总共包含628个待训练参数，相比其他DNN模型少了很多。

```python
afmal.model.fit(train_dataset, epochs=20)

#output:
Train for 199 steps
Epoch 1/20
199/199 [==============================] - 2s 11ms/step - loss: 0.5290 - binary_accuracy: 0.7166 - recall: 0.1871
Epoch 2/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4698 - binary_accuracy: 0.7520 - recall: 0.3333
Epoch 3/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4562 - binary_accuracy: 0.7687 - recall: 0.4497
Epoch 4/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4458 - binary_accuracy: 0.7764 - recall: 0.4947
Epoch 5/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4410 - binary_accuracy: 0.7782 - recall: 0.5105
Epoch 6/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4378 - binary_accuracy: 0.7840 - recall: 0.5222
Epoch 7/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4348 - binary_accuracy: 0.7886 - recall: 0.5398
Epoch 8/20
199/199 [==============================] - 1s 4ms/step - loss: 0.4323 - binary_accuracy: 0.7919 - recall: 0.5310
Epoch 9/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4258 - binary_accuracy: 0.7965 - recall: 0.5444
Epoch 10/20
199/199 [==============================] - 1s 4ms/step - loss: 0.4267 - binary_accuracy: 0.7969 - recall: 0.5450
Epoch 11/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4282 - binary_accuracy: 0.7982 - recall: 0.5579
Epoch 12/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4196 - binary_accuracy: 0.7974 - recall: 0.5532
Epoch 13/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4205 - binary_accuracy: 0.7979 - recall: 0.5538
Epoch 14/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4165 - binary_accuracy: 0.7987 - recall: 0.5456
Epoch 15/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4151 - binary_accuracy: 0.7999 - recall: 0.5433
Epoch 16/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4156 - binary_accuracy: 0.8017 - recall: 0.5363
Epoch 17/20
199/199 [==============================] - 1s 4ms/step - loss: 0.4145 - binary_accuracy: 0.8037 - recall: 0.5485
Epoch 18/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4177 - binary_accuracy: 0.8007 - recall: 0.5409
Epoch 19/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4141 - binary_accuracy: 0.8021 - recall: 0.5415
Epoch 20/20
199/199 [==============================] - 1s 5ms/step - loss: 0.4139 - binary_accuracy: 0.8039 - recall: 0.5392
<tensorflow.python.keras.callbacks.History at 0x1a7c092eda0>
```

20轮迭代后，准确率在80%左右，召回率差不多54%，和其他DNN+FM模型差不多，但是由于参数少，训练时间大大减小，这里就不介绍调参工作了。

### 3.4 模型评估
```python
loss, acc, recall = afmal.model.evaluate(test_dataset)

#output:
23/23 [==============================] - 0s 11ms/step - loss: 0.3909 - binary_accuracy: 0.7986 - recall: 0.4843
```

验证集上准确率为79.86%，和训练集差不多，召回率为48.43%。

## 4 小结
相比于其他的DNN模型（比如Wide&Deep，Deep&Cross）都是通过MLP来隐式学习组合特征，AttentionFM是在FM的基础上改进的，通过两个隐向量内积来学习组合特征，可解释性更好。

通过直接扩展FM，AFM引入Attention机制来学习不同组合特征的权重，即保证了模型的可解释性又提高了模型性能（但个人觉得这里的缺点是使用了物理意义并不明显的哈达玛积，笔者认为可以采用self-Attention试试）。

DNN的另一个作用是提取高阶组合特征，而AttentionFM在交叉项加权累加后直接与一阶项相加作为输出，并没有进行更深的网络去学习高阶交叉特征，这可能是AttentionFM模型的一个缺点。
## 1 什么是FFM？
**FFM**全称“Field-aware Factorization Machine”， 顾名思义是FM（Factorization Machine）模型的升级版，在FM的基础上引入field的概念，把相同性质的特征归于同一个field，将特征所在的不同的field 这个信息也考虑进去，从而进行特征组合的二分类模型。  

## 2 FFM有什么用？
简单来说，FFM和FM目的一样，是为了解决**特征大规模稀疏场景下的特征交叉组合问题**，主要应用在推荐系统的排序阶段。在传统的线性模型中，每个特征都是独立的，如果需要考虑特征与特征之间的相互作用，可能需要人工对特征进行交叉组合。非线性SVM可以对特征进行核变换，但是在特征高度稀疏的情况下，并不能很好的进行学习。由于推荐系统的场景就是特征的大规模稀疏，由此产生了FM系列算法，包括FM，FFM，DeepFM等算法。  
## 3 FFM模型原理
在介绍FFM模型原理之前，先回顾一下FM模型的原理：
$$
y = w_0 + \sum_{i=1}^{n}{w_i x_i} + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_i,v_j \gt x_i x_j}
$$
其中，$v\in R^{n\times k}$，对于每个特征都有一个长度为k的隐向量。$\lt v_i,v_j \gt$表示的是两个长度为k的向量的点积。  
$$
\lt v_i,v_j \gt = \sum_{f=1}^{k}{v_{i,f} \cdot v_{j,f}}
$$
FFM则是将隐向量v又进一步细化，引入 field 概念，将特征所在的field 信息也考虑进去。公式表达如下所示：  
$$
y = w_0 + \sum_{i=1}^{n}{w_i x_i} + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}{\lt v_{i,f_j},v_{j,f_i} \gt x_i x_j}
$$

其中，n是特征数量，$f_j$是第$ j$个特征所属的field。由于隐向量$v$的长度为$k$，FFM的二次参数有$nfk$个，远多于 FM 模型的 $nk$个。FFM与FM的区别在于隐向量由原来的$v_i$变成了$v_{i,f_j}$，这意味着每个特征对应的不是唯一的一个隐向量，而是一组隐向量。当特征$x_i$与特征$x_j$进行交叉时，特征$x_i$会从$x_i$的一组隐向量中选择出与特征$x_j$的域$f_j$对应的隐向量$v_{i,f_j}$进行交叉。同理，$x_j$也会选择与$x_i$的域$f_i$对应的隐向量$v_{j,f_i}$进行交叉。此外，由于隐向量与field相关，FFM二次项并不能够化简，其预测复杂度是$O(kn^2)$。

下面以一个例子简单说明FFM的特征组合方式。某个输入样本记录如下：  

|  User  |  Movie  |     Genre     | Price |
| :----: | :-----: | :-----------: | :---: |
| YuChin | 3Idiots | Comedy, Drama | $9.99 |

这条记录可以编码成5个特征，其中“Genre=Comedy”和“Genre=Drama”属于同一个field，“Price”是数值型，简单起见，这里不做编码转换。为了方便说明FFM的样本格式，我们将所有的特征和对应的field映射成整数编号。

| Field name | Field index | Feature name  | Feature index |
| :--------: | :---------: | :-----------: | :-----------: |
|    User    |    **1**    |  User=YuChin  |     **1**     |
|   Movie    |    **2**    | Movie=3Idiots |     **2**     |
|   Genre    |    **3**    | Genre=Comedy  |     **3**     |
|   Price    |    **4**    |  Genre=Drama  |     **4**     |
|            |             |     Price     |     **5**     |

这里有5个特征，考虑二姐特征组合就有$C_5^2=10$个组合，如下所示：  
$$
⟨v_{1,2},v_{2,1}⟩⋅1⋅1+⟨v_{1,3},v_{3,1}⟩⋅1⋅1+⟨v_{1,3},v_{4,1}⟩⋅1⋅1+⟨v_{1,4},v_{5,1}⟩⋅1⋅9.99 +⟨v_{2,3},v_{3,2}⟩⋅1⋅1+⟨v_{2,3},v_{4,2}⟩⋅1⋅1+⟨v_{2,4},v_{5,2}⟩⋅1⋅9.99 +⟨v_{3,3},v_{4,3}⟩⋅1⋅1+⟨v_{3,4},v_{5,3}⟩⋅1⋅9.99 +⟨v_{4,4},v_{5,3}⟩⋅1⋅9.99
$$

## 4 损失函数

简单起见，这里忽略常数项和一阶项，模型公式如下：  
$$
\phi (x) = \sum_{j_1,j_2 \in C_2}<w_{j_1,f_2}, w_{j_2,f_1}>x_{j_1}x_{j_2}
$$
其中， $C_2$是非零特征的二元组合，j_1是特征，属于field $f_1$，$w_{j_1,f_2}$ 是特征 $j_1$ 对field $f_2$ 的隐向量。此FFM模型采用logistic loss作为损失函数，和L2惩罚项，因此只能用于二元分类问题。

$$
P(y=1|x)=\frac {e^{\phi (x)}}{1+e^{\phi (x)}}=1-\sigma (\phi (x))
$$

$$
P(y=0|x)=\frac {1}{1+e^{\phi (x)}}=\sigma (\phi (x))
$$

似然函数：
$$
\prod_{i=1}^{n} {\sigma (\phi (x_i))^{y_i}} \cdot {(1- \sigma (\phi (x_i)))^{1- y_i}}
$$
最大化对数似然函数：
$$
max \quad \sum_{i=1}^{n} {[y_i(\sigma (\phi (x_i)))+(1-y_i)(1- \sigma (\phi (x_i)))]}
$$
加L2正则后，损失函数如下：
$$
L(w) = -\sum_{i=1}^{n} {[y_i(\sigma (\phi (x_i)))+(1-y_i)(1- \sigma (\phi (x_i)))]} + \frac{\lambda}{2} \cdot ||w||^2
$$

## 5 基于tensorflow2.0的ffm模型实践

这里使用的是Kaggle比赛——Telco Customer Churn电信客户流失数据集。

### 5.1 导入包

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from ffm import FFMLayer, FieldawareFactorizationMachine

# 设置GPU显存动态增长
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
```

### 5.2 ffm实现细节

ffm.py文件

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class FFMLayer(keras.layers.Layer):
    def __init__(self, factor_dim, field_dict, **kwargs):
        self.factor_dim = factor_dim
        self.field_dict = field_dict
        self.n_features = len(field_dict)
        self.field_dim = len(set(field_dict.values()))
        super(FFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(self.n_features, self.field_dim, self.factor_dim),
                                 trainable=True,
                                 initializer='random_uniform')

        super(FFMLayer, self).build(input_shape)

    def call(self, x):
        interaction_term = tf.zeros(shape=(1,), dtype=tf.float32)
        for i in range(self.n_features - 1):
            for j in range(i + 1, self.n_features):
                wij = tf.reduce_sum(
                    tf.math.multiply(self.w[i, self.field_dict[j], :], self.w[j, self.field_dict[i], :]))
                interaction_term += wij * tf.math.multiply(x[:, i], x[:, j])
        interaction_term = tf.reshape(interaction_term, [-1, 1])
        return interaction_term


class FieldawareFactorizationMachine:
    def __init__(self, factor_dim, field_dict, n_features):
        self.factor_dim = factor_dim
        self.field_dict = field_dict
        self.x_in = keras.Input(shape=(n_features,), name="inp")
        self.lr = keras.layers.Dense(1, use_bias=True)
        self.ffm_layer = FFMLayer(factor_dim, field_dict)
        self.activate = keras.layers.Activation("sigmoid")

    def build(self):
        lr_term = self.lr(self.x_in)
        interaction_term = self.ffm_layer(self.x_in)
        self.out = lr_term + interaction_term
        self.out = self.activate(self.out)
        self.model = keras.Model(self.x_in, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer='adam',
                           metrics=['acc'])
```

### 5.3 数据处理

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
features = data.columns[2:-1]
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

dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = dataset.shuffle(len(X_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
test_dataset = test_dataset.batch(32)

X_train.shape, X_test.shape

# output:
((6338, 31), (705, 31))
```

### 5.4 模型训练

```python
# 隐向量长度
factor_dim = 8
ffmal = FieldawareFactorizationMachine(factor_dim, feild_dict, len(features))
ffmal.build()
ffmal.model.fit(train_dataset, epochs=20)

#output:
Train for 199 steps
Epoch 1/20
199/199 [==============================] - 29s 145ms/step - loss: 0.5826 - acc: 0.6925
Epoch 2/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4399 - acc: 0.7881
Epoch 3/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4324 - acc: 0.7944
Epoch 4/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4242 - acc: 0.8014
Epoch 5/20
199/199 [==============================] - 8s 40ms/step - loss: 0.4226 - acc: 0.8029
Epoch 6/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4191 - acc: 0.8044
Epoch 7/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4192 - acc: 0.8042
Epoch 8/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4151 - acc: 0.8025
Epoch 9/20
199/199 [==============================] - 8s 40ms/step - loss: 0.4157 - acc: 0.8050
Epoch 10/20
199/199 [==============================] - 8s 40ms/step - loss: 0.4126 - acc: 0.8078
Epoch 11/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4117 - acc: 0.8077
Epoch 12/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4223 - acc: 0.8085
Epoch 13/20
199/199 [==============================] - 8s 40ms/step - loss: 0.4134 - acc: 0.8088
Epoch 14/20
199/199 [==============================] - 8s 40ms/step - loss: 0.4093 - acc: 0.8099
Epoch 15/20
199/199 [==============================] - 8s 40ms/step - loss: 0.4149 - acc: 0.8105
Epoch 16/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4112 - acc: 0.8099
Epoch 17/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4081 - acc: 0.8102
Epoch 18/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4082 - acc: 0.8107
Epoch 19/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4096 - acc: 0.8099
Epoch 20/20
199/199 [==============================] - 8s 41ms/step - loss: 0.4101 - acc: 0.8105
```

可以看到训练集上最终准确率在81%左右。

### 5.5 模型评估

验证集上准确率是79%，效果不错。

```python
loss, acc = fm_al.model.evaluate(test_dataset)
#损失、准确率
loss, acc

#output
23/23 [==============================] - 6s 239ms/step - loss: 0.4015 - acc: 0.7957
(0.4015362325893796, 0.79574466)
```
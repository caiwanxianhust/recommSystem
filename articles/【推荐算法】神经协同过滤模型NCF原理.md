## 1 算法背景

本文要介绍的基于神经网络的协同过滤算法（NCF）来自由新加坡国立大学、哥伦比亚大学以及山东大学于2017年发表的文章[《Neural Collaborative Filtering》](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)，文章提出**基于神经网络的技术来解决推荐中的关键问题-基于隐式反馈的协同过滤**。

在信息爆炸的时代，推荐系统在减轻信息过载方面发挥了巨大的作用，被众多在线服务，包括电子商务，网络新闻和社交媒体等广泛采用。个性化推荐系统的关键在于根据过去用户交互的内容（e.g. 评分、点击等行为），对用户对项目的偏好进行建模，就是所谓的协同过滤。在众多的协同过滤技术中，矩阵分解（MF）是最受欢迎的，它将用户和项目映射到共享潜在空间（shared latent space），使用隐向量（latent features），用以表示用户或项目。这样一来，用户在项目上的交互就被建模为它们隐向量之间的内积。关于矩阵分解这里不做详细介绍，具体可参考笔者的另一篇文章[《【推荐算法】基于矩阵分解的协同过滤算法》](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247483693&idx=3&sn=4f4b394394a96eea34fbbeabc9b5342d&chksm=e9278494de500d82cb41bdecd4099c36ac1d66f4f2896f6143c2913e9c08f3804614956b16df&token=1195106363&lang=zh_CN#rd)。

矩阵分解（MF）直接使用隐向量之间的内积作为预测结果，简单有效，但简单地将特征隐向量的乘积线性组合的方式可能不足以捕捉复杂的用户交互信息，因此模型表达能力受限。

如何突破这个限制？设计一个比内积更好的专用交互函数，用于建模用户和项目之间的隐藏特征交互即可。原文用神经网络结构代替内积，提出了一个通用框架NCF。NCF是泛型的，在其框架下可以表示和推广矩阵分解。为了加强非线性NCF建模，利用多层感知器来学习用户-项目交互功能。

## 2 模型结构

原文提出的通用框架如下：

![ncf通用框架](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5o8GrcSSwoKKstW2cleghyYIjibDNvtcKcObhLboY84OhKNnhhzkqa914mAibXar9ZfneibQO6bw5LSg/0?wx_fmt=png)

输入层上面是嵌入层（Embedding Layer）;它是一个全连接层，用来将输入层的稀疏表示映射为一个稠密向量（dense vector）。所获得的用户（项目）的嵌入（就是一个稠密向量）可以被看作是在潜在因素模型的上下文中用于描述用户（项目）的隐向量。然后将用户向量和项目向量输入多层神经网络结构，把这个结构称为神经协同过滤层（NeuralCF Layer），它将隐向量映射为预测分数。NCF层的每一层可以被定制，用以发现用户-项目交互的某些隐藏结构。最终输出层是预测分数 $\hat y_{ui}$，通过最小化 $\hat y_{ui}$和其目标值$y_{ui}$间的损失函数进行模型训练。

这里给出NCF模型的预测公式：
$$
\hat y_{ui} = f(P^T v^U_u,Q^T v^I_i | P,Q,\Theta _f)
$$
式中$P \in R^{M\times K}$和$Q \in R^{N \times K}$分别表示用户和项目的隐向量矩阵（$M$个用户，$N$个项目，向量长度为$K$）；$\Theta _f$表示交互函数$f$的参数；交互函数$f$被定义为多层神经网络，可以表示为：
$$
f(P^T v^U_u,Q^T v^I_i)=\phi_{out}(\phi_{X}(...\phi_2(\phi_1(P^T v^U_u,Q^T v^I_i))...))
$$
其中$\phi _{out}$和$\phi_X$分别表示为输出层和第$X$个神经协同过滤（CF）层映射函数，总共有$X$个神经协同过滤（CF）层。

模型损失函数对于评分预测的回归任务直接使用均方误差（MSE），对于是否感兴趣这种分类任务可以采用交叉熵损失函数，此外为了削弱不平衡样本集带来的影响可以加入样本权重参数，这里就不多介绍了。

针对这个NCF通用框架，原文提出了三种不同的实现，分别为**广义矩阵分解（GMF）、多层感知机（MLP）和神经矩阵分解（NeuMF）**，三种实现可以用一张图来说明：

![ncf模型结构](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5o8GrcSSwoKKstW2cleghyY3VSNI1m2dOAmcMPT3wWhmUibzg8geHmYRVh7f47dYSWxico1SdQd3rhQ/0?wx_fmt=png)

### 2.1 广义矩阵分解（GMF）

上图中仅使用GMF layer，就得到了第一种实现方式GMF，GMF被称为广义矩阵分解，计算公式为：
$$
\hat y_{ui} = a_{out}(h^T (p_u \odot q_i))
$$
式中$a_{out}$表示输出层的激活函数，如果是分类通常是$sigmoid$或者$softmax$，如果是回归通常不需要激活；$p_u \odot q_i$表示用户隐向量和项目隐向量按元素乘积，是这个模型的核心。

这里给出笔者的实现，比较简单不再赘言。

```python
self.user_embedding_gmf = keras.layers.Embedding(num_users + 1, emb_size, name="user_emb_gmf")
self.item_embedding_gmf = keras.layers.Embedding(num_items + 1, emb_size, name="item_emb_gmf")
self.multiply_gmf = keras.layers.Multiply(name="multiply_gmf")
self.reshape_layer = keras.layers.Reshape((emb_size,), name="reshape")

# (batch_size, 1) -> (batch_size, 1, emb_size)
user_emb_gmf = self.user_embedding_gmf(self.user_inp)
# (batch_size, 1, emb_size) -> (batch_size, emb_size)
user_emb_gmf = self.reshape_layer(user_emb_gmf)
# (batch_size, 1) -> (batch_size, 1, emb_size)
item_emb_gmf = self.item_embedding_gmf(self.item_inp)
# (batch_size, 1, emb_size) -> (batch_size, emb_size)
item_emb_gmf = self.reshape_layer(item_emb_gmf)
# (batch_size, emb_size)
self.gmf_out = self.multiply_gmf([user_emb_gmf, item_emb_gmf])
```

### 2.2 多层感知机（MLP）

上图中仅使用右侧的MLP Layers，就得到了第二种实现方式MLP，通过多层神经网络来学习user和item的隐向量。这样，输出层的计算公式为：
$$
z_1 = \phi_1(p_u,q_i)=
\begin{pmatrix}
p_u\\
q_i
\end{pmatrix}
$$

$$
\phi_2(z_1) = a_2(W^T_2 z_1 + b_2)
$$

$$
...
$$

$$
\phi_L(z_{L-1}) = a_L(W^T_L z_{L-1} + b_L)
$$

$$
\hat y_{ui} = \sigma (h^T \phi _L (z_{L-1}))
$$

简单来说就是把用户隐向量和项目隐向量拼接起来，然后用多层神经网络预测结果即可，下面给出笔者的实现：

```python
self.user_embedding_mlp = keras.layers.Embedding(num_users + 1, emb_size, name="user_emb_mlp")
self.item_embedding_mlp = keras.layers.Embedding(num_items + 1, emb_size, name="item_emb_mlp")
self.concate_emb = keras.layers.Concatenate(axis=-1, name="concate_emb")
self.reshape_layer = keras.layers.Reshape((emb_size,), name="reshape")
self.mlp_1 = keras.layers.Dense(emb_size * 2, activation="relu")
self.mlp_2 = keras.layers.Dense(emb_size, activation="relu")
self.mlp_3 = keras.layers.Dense(emb_size, activation="relu")
self.dropout_1 = keras.layers.Dropout(rate=drop_rate)
self.dropout_2 = keras.layers.Dropout(rate=drop_rate)
self.dropout_3 = keras.layers.Dropout(rate=drop_rate)

# (batch_size, 1) -> (batch_size, 1, emb_size)
user_emb_mlp = self.user_embedding_mlp(self.user_inp)
# (batch_size, 1, emb_size) -> (batch_size, emb_size)
user_emb_mlp = self.reshape_layer(user_emb_mlp)
# (batch_size, 1) -> (batch_size, 1, emb_size)
item_emb_mlp = self.item_embedding_mlp(self.item_inp)
# (batch_size, 1, emb_size) -> (batch_size, emb_size)
item_emb_mlp = self.reshape_layer(item_emb_mlp)
# (batch_size, emb_size) + (batch_size, emb_size) -> (batch_size, emb_size * 2)
interaction = self.concate_emb([user_emb_mlp, item_emb_mlp])
# (batch_size, emb_size * 2) -> (batch_size, emb_size * 2)
mlp_out = self.mlp_1(interaction)
mlp_out = self.dropout_1(mlp_out)
# (batch_size, emb_size * 2) -> (batch_size, emb_size)
mlp_out = self.mlp_2(mlp_out)
mlp_out = self.dropout_2(mlp_out)
# (batch_size, emb_size) -> (batch_size, emb_size)
mlp_out = self.mlp_3(mlp_out)
self.mlp_out = self.dropout_3(mlp_out)
```


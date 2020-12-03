## 1 算法背景

本文要介绍的基于神经网络的协同过滤算法（NCF）来自由新加坡国立大学、哥伦比亚大学以及山东大学于2017年发表的文章[《Neural Collaborative Filtering》](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)，文章提出**基于神经网络的技术来解决推荐中的关键问题-基于隐式反馈的协同过滤**。

在信息爆炸的时代，推荐系统被众多在线服务，包括电子商务，网络新闻和社交媒体等广泛采用，极大的缓解了互联网信息过载的问题。个性化推荐系统的关键在于根据用户交互历史（e.g. 评分、点击等行为），对用户对项目的偏好进行建模，就是所谓的协同过滤。在众多的协同过滤模型中，矩阵分解（MF）将用户和项目映射到共享潜在空间（shared latent space），使用隐向量（latent features）表示用户或项目，这样一来，用户在项目上的交互就被建模为它们隐向量之间的内积，简单高效，受到一致青睐。关于矩阵分解这里不做详细介绍，具体可参考笔者的另一篇文章[《【推荐算法】基于矩阵分解的协同过滤算法》](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247483693&idx=3&sn=4f4b394394a96eea34fbbeabc9b5342d&chksm=e9278494de500d82cb41bdecd4099c36ac1d66f4f2896f6143c2913e9c08f3804614956b16df&token=1195106363&lang=zh_CN#rd)。

矩阵分解（MF）直接使用隐向量之间的内积作为预测结果，简单有效，但简单地将特征隐向量的乘积线性组合的方式可能不足以捕捉复杂的用户交互信息，因此模型表达能力受限。

如何突破这个限制？设计一个比内积更好的专用交互函数，用于建模用户和项目之间的隐藏特征交互即可。原文用神经网络结构代替内积，提出了一个通用框架NCF，在其框架下既可以使用矩阵分解，同时为了加强非线性NCF建模，又可以利用多层感知器来学习用户-项目交互功能。

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
式中$a_{out}$表示输出层的激活函数，如果是分类通常是$sigmoid$或者$softmax$，如果是回归通常不需要激活；$p_u$和$q_i$表示用户和项目的嵌入向量；$p_u \odot q_i$表示用户隐向量和项目隐向量按元素乘积，是这个模型的核心，其本质是应用了一个线性内核来模拟隐藏特征交互。

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

GMF模型结构图如下：

```python
self.gmf_model = keras.Model([self.user_inp, self.item_inp], self.gmf_out, name="gmf")
keras.utils.plot_model(ncfal.gmf_model, 'ncf_gmf_model.png', show_layer_names=True, show_shapes=True)
```

![gmf模型结构图](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qIiaYhD0QAsJibEZM2G4ehfwFicYUxJKrNWvp48vkd8CfwnRhdGUPUjEIM4WHgnA97l65O3VtEc76ibw/0?wx_fmt=png)

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

式中$a_X$表示全连接层的激活函数，这里使用ReLU，$\phi _X$表示第$X$层的输出。

简单来说MLP模型就是把用户隐向量和项目隐向量拼接起来，然后用多层神经网络预测结果，其本质是使用非线性内核从数据中学习交互函数。

下面给出笔者的实现：

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

MLP模型结构图如下：

```python
self.mlp_model = keras.Model([self.user_inp, self.item_inp], self.mlp_out, name="mlp")
keras.utils.plot_model(ncfal.mlp_model, 'ncf_mlp_model.png', show_layer_names=True, show_shapes=True)
```

![mlp模型结构图](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qIiaYhD0QAsJibEZM2G4ehfw0Eob4UIW1d4r4dRicS8O6w4uEmibxqG9SQF4EuEchWlJ1guibLPibqjSVA/0?wx_fmt=png)

### 2.3 NeuMF

将GMF和MLP两者的输出拼接后，得到的就是第三种实现方式，为了使得融合模型具有更大的灵活性，可以在GMF和MLP学习独立的嵌入向量，并结合两种模型通过连接他们最后的隐层输出。上图是该方式的完整实现，完整的计算公式为：
$$
\phi ^{GMF} = p^G_u \odot q^G_i
$$

$$
\phi ^{MLP} = a_L(W^T_L(a_{L-1}(... a_2(W^T_2 
\begin{pmatrix}
p_u^M\\
q_i^M
\end{pmatrix}
+ b_2)...))+b_L)
$$

$$
\hat y_{ui} = \sigma (h^T 
\begin {pmatrix}
\phi ^{GMF}\\
\phi ^{MLP}
\end {pmatrix}
)
$$

式中$p^G_u$和$p^M_u$分别代表 GMF 部分和 MLP 部分的用户嵌入（user embedding）；同样的，$q^G_i$和$q^G_i$分别表示项目的嵌入（item embedding）。使用ReLU函数作为 MLP层的激活函数，该模型结合MF的线性度和DNNs的非线性度，用以建模用户-项目之间的潜在结构。我们将这一模式称为“NeuMF”，简称神经矩阵分解（Neural Matrix Factorization）。

下面给出笔者的实现。

```python
self.concate_com = keras.layers.Concatenate(axis=-1, name="concate_com")
self.com_dense = keras.layers.Dense(1, activation="sigmoid", name="com_dense")

# (batch_size, emb_size) + (batch_size, emb_size) -> (batch_size, emb_size * 2)
self.com_out = self.concate_com([self.gmf_out, self.mlp_out])
# (batch_size, emb_size * 2) -> (batch_size, 1)
self.com_out = self.com_dense(self.com_out)
```

NeuMF模型结构图如下：

```python
self.neumf_model = keras.Model([self.user_inp, self.item_inp], self.com_out, name="neumf")
keras.utils.plot_model(ncfal.neumf_model, 'ncf_neumf_model.png', show_layer_names=True, show_shapes=True)
```

![NeuMF模型结构图](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qIiaYhD0QAsJibEZM2G4ehfwjkvehcib4aAFyiayMic3xn3ibY3gOpJkdS3G3ZLrlZsrRlGsLVz6liaO7Hw/0?wx_fmt=png)

## 3 模型训练

### 3.1 数据集

原文选用了两个公开的数据集[MovieLens](http://grouplens.org/datasets/movielens/1m/)和 [Pinterest](https://sites.google.com/site/xueatalphabeta/academic-projects)，这两个数据集被广泛用于协同过滤模型评测，非常具有代表性。

- **MovieLens**：这个电影评级数据集被广泛地用于评估协同过滤算法。原文使用的是包含一百万个评分的版本，每个用户至少有20个评分（评分为1~5）。 虽然这是显性反馈数据集，但原文有意选择它来挖掘NCF模型从显式反馈中学习隐性信号的表现。为此，原文将其转换为隐式数据，其中每个评分被标记为0或1表示用户是否已对该项进行评级。
- **Pinterest**：这个隐含的反馈数据的构建用于评估基于内容的图像推荐。原始数据非常大但是很稀疏。 例如，超过20％的用户只有一个pin（pin类似于点赞），使得难以用来评估协同过滤算法。 因此，原文使用与MovieLens数据集相同的方式过滤数据集，即仅保留至少有过20个pin的用户。处理后得到了包含55187个用户和1580809个项目交互的数据的子集。 每个交互都表示用户是否将图像pin在自己的主页上。

### 3.2 评估方案

为了评价项目推荐的性能，采用leave-one-out方法评估，即对于每个用户，将其最近的一次交互作为测试集（按交互行为的时间戳取最近一次交互行为），并利用余下的数据作为训练集。由于在评估过程中为每个用户排列所有项目花费的时间太多，所以随机抽取100个不与用户进行交互的项目，将测试项目排列在这100个项目中。排名列表的性能由**命中率（HR）**和**归一化折扣累积增益（NDCG）**来衡量，取前10名。如此一来，HR直观地衡量测试项目是否存在于前10名列表中，而NDCG通过将较高分数指定为顶级排名来计算命中的位置。

### 3.3 超参数设置

- 可以使用网格搜索交叉验证的方法来进行超参数选取。
- 采用高斯分布随机初始化模型参数，用Adam优化器进行模型优化，损失函数采用交叉熵损失函数。
- batch_size、learning_rate、hidden_size等等这些超参数都可以根据在验证集上的表现做相应调整。

## 4 小结

原文设计了一个通用框架NCF，并提出了三种实例：GMF，MLP和NeuMF，以不同的方式模拟用户-项目交互。这项工作补充了主流浅层协同过滤模型，为深入学习推荐研究开辟了新途径。


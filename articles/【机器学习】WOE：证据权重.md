**写在前面**：这篇文章昨天已经发布过一次，今天重读，发现之前还是有一些理解错误的地方，由于指数函数在0附近的趋势近似一条直线，而本文中的数据集算出来的age特征的WOE值恰巧都在0附近，给笔者造成了错觉，错误地认为WOE和Bad Rate成线性关系，修改后笔者对这部分逻辑进行了简单的推导。

## 1 为什么需要WOE？

在风控模型所使用的数据里，我们会用到两种变量类型：  
- Numerical Variable，数值变量。例如逾期金额，天数、客户年龄等。  
- Categorical Variable，类别变量。例如客户职业、客户性别、婚姻状况等。  

在制作评分卡过程中，我们有时需要把数值变量变成类别变量，例如客户年龄段，我们可以划分为 `[20及以下],[21-30],[31-40],[41-50],[51-60],[61-70],[70以上]` 七个类别，这时候我们就把数值变成了类别。这种把数值变成类别的技巧叫做分箱（binning）。  

但是当把所有变量都变成类别后，怎么去训练一个模型呢？例如逻辑回归，只能用数值作为特征输入。怎么把类别转换成数值呢？  

这时候首先想到的可能是 one-hot 编码（或者dummy），但 one-hot 编码并不是一种良好的分类变量编码方法，它的特性会使得特征维度迅速膨胀，导致结果异常稀疏，但实际上并没有太多有用的信息，增加计算量的同时模型也很难有很好的效果。  

这时候，我们可以试试把类别或者分箱转化成相应的数值（纳尼，不是刚把数值变成分箱么？？？）。这个数值最好有这个特性：分数越大，代表这个变量给bad label的贡献度越大，这个贡献度，视运算符号不同，可以是正向，也可以是负向，但我们期望它们之间有个单调关系，因为我们的模型通常是逻辑回归这类线性模型。  

这时候我们需要引入WOE编码。  
## 2 什么是WOE？
WOE全称是Weight of Evidence，即证据权重，是广泛用于风控模型的一种编码方式。
WOE的公式定义如下：  
$$
\begin {equation}
WOE_i = \ln (\frac{\frac{Bad_i}{Good_i}}{\frac{Bad_{total}}{Good_{total}}}) = \ln \frac {Bad_i}{Good_i} - \ln \frac {Bad_{total}}{Good_{total}}
\label {eq:1.1}
\end {equation}
$$
其中，`i` 为待编码变量的第 `i` 个取值（或者说分箱），$Bad_i$ 为第 `i` 个取值（或者说分箱）中坏样本的数量，$Bad_{total}$ 为总样本中坏样本的数量，$Good_i$ 与 $Good_{total}$ 的意义同理。

- WOE 可以理解为**当前组中正负样本的比值，与所有样本中正负样本比值的差异**。这个差异是用这两个比值的比值，再取对数来表示的。
- WOE > 0 表示当前组正负样本比例大于总体的正负样本比例，值越大表示这个分组里的坏样本的可能性越大；
- WOE < 0 表示当前组正负样本比例小于总体的正负样本比例，值越小表示这个分组里的坏样本的可能性越小。
- WOE绝对值越大，对于分类贡献越大。当分箱中正负的比例等于随机（大盘）正负样本的比值时，说明这个分箱没有预测能力，即 WOE = 0。

## 3 怎么计算WOE？

**导入包**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
import seaborn as sns

%matplotlib inline
sns.set_style("darkgrid",{"font.sans-serif":['SongTi', 'Times New Roman']})   #这是方便seaborn绘图得时候得字体设置
plt.rcParams['savefig.dpi'] = 720 #图片像素
plt.rcParams['figure.dpi'] = 720 #分辨率
```

**读取数据并进行变量命名**

数据集中每一行数据用空格符号隔开，总共1000行21列。

```python
df = pd.read_csv("./data/german.csv", header=None, delim_whitespace=True)
columns = ['status_account', 'duration', 'credit_history', 'purpose', 'amount','svaing_account', 'present_emp', 
           'income_rate', 'personal_status', 'other_debtors', 'residence_info', 'property', 'age', 'linst_plans',
           'housing', 'num_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'target']
df.columns = columns
# 将标签减1转化为0、1。其中，0表示好用户，没有发生违约；1表示坏用户，发生了违约。
df['target'] -= 1
```

**坏样本比例**

```python
df['target'].value_counts().plot.pie(autopct='%.2f%%', title='label distribution')
```

<img src="https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5ptJrUNuZjT5olag7FASvt3OUiaKfm7TAePQFfb8Ym3T70dBJMukUyAzTKbe49hn8ehUDrsVlR4BFw/0?wx_fmt=png" style="zoom: 33%;" />

**客户的年龄分布**

```python
plt.figure(figsize=(12,8))
age_plot = sns.distplot(df['age'])
age_plot.set_title("Age Distribuition", fontsize=18)
age_plot.set_xlabel("age")
age_plot.set_ylabel("Probability", fontsize=15)
```

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5ptJrUNuZjT5olag7FASvt3ibRZjzeOEGRFHMiaLzSBbicWxd7gEHNNReguKFvIydRdaYFJ0ZKKMKZSA/0?wx_fmt=png)

**对年龄这个特征进行分箱**

 0-20 Teenager， 21-40 Young， 41-55 Middle， 56-120 Old

```python
bin_labels = ['Teenager','Young','Middle','Old']
bin_edges = [0,20,40,55,120]
df['age_group'] = pd.cut(df['age'], bin_edges, labels=bin_labels) 
```

**看一下年龄分箱后好坏样本分布**

```python
def percentage_above_bar_relative_to_xgroup(ax):
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    for bars in ax.containers:
        for i, p in enumerate(bars):
            total = sum(xgroup[i] for xgroup in all_heights)
            percentage = '{}%'.format(round(p.get_height() / total * 100, 1))
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), size=11, ha='center', va='bottom')
            
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='age_group', data=df, hue='target', ax=ax)
percentage_above_bar_relative_to_xgroup(ax)
ax.set(xlabel='age_group', ylabel='count')
```

<img src="https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rtx3P64VmhR4DAic0pibdAIkhFeEEW2UWciahMNG6SZV9blFfXtcMyHibKQdeDf3mvFUL1K69vnduIsg/0?wx_fmt=png" style="zoom:50%;" />

**统计各个箱中好坏样本数量**

```python
# 统计各个箱中好坏样本数量
woe_bad_cnt = df[df['target'] == 1].groupby('age_group')['target'].count().reset_index()
woe_good_cnt = df[df['target'] == 0].groupby('age_group')['target'].count().reset_index()
# 连接两个df
woe_bad_good = pd.merge(woe_bad_cnt, woe_good_cnt, 
         how='inner', on='age_group', 
         suffixes=('_bad_cnt', '_good_cnt'))
woe_bad_good

# output:
	age_group	target_bad_cnt	target_good_cnt
0	Teenager	6	10
1	Young	    222	488
2	Middle	    53	150
3	Old	        19	52
```

**计算各个箱中好坏样本数量占总样本的比例**

```python
woe_bad_good['bad_ratio'] = woe_bad_good['target_bad_cnt'] / df[df['target'] == 1]['target'].count()
woe_bad_good['good_ratio'] = woe_bad_good['target_good_cnt'] / df[df['target'] == 0]['target'].count()
woe_bad_good

#output:
	age_group	target_bad_cnt	target_good_cnt	bad_ratio	good_ratio
0	Teenager	6	10	0.020000	0.014286
1	Young	    222	488	0.740000	0.697143
2	Middle	    53	150	0.176667	0.214286
3	Old	        19	52	0.063333	0.074286
```

**计算WOE值**

```python
woe_bad_good['woe'] = np.log(woe_bad_good["bad_ratio"]) - np.log(woe_bad_good["good_ratio"])
woe_bad_good

#ouput:
	age_group	target_bad_cnt	target_good_cnt	bad_ratio	good_ratio	woe
0	Teenager	6	10	0.020000	0.014286	0.336472
1	Young	    222	488	0.740000	0.697143	0.059660
2	Middle	    53	150	0.176667	0.214286	-0.193046
3	Old	        19	52	0.063333	0.074286	-0.159507
```

可以看到 `Teenager，Young，Middle，Old` 这4个取值对应的WOE值分别为 `0.336472,0.059660,-0.193046,-0.159507`。

现在有了年龄段变量各个取值对应的WOE值了，怎么将数据集转换成WOE呢？很简单，直接根据特征apply-map，或者使用pd.merge将前后两个DataFrame连接起来也可以，这里代码如下：

```python
df = pd.merge(df, woe_bad_good[["age_group", "woe"]], how='inner', on='age_group')
df[["age", "target", "age_group", "woe"]].head()

#output:
	age	target	age_group	woe
0	67	0	Old	-0.159507
1	61	0	Old	-0.159507
2	60	1	Old	-0.159507
3	63	1	Old	-0.159507
4	57	0	Old	-0.159507
```

## 4 什么情况下使用WOE？

说了这么多，这里有个疑问：类别变量我能理解要WOE，但对于连续变量，通过分箱变成类别变量，又通过WOE去转换连续变量，这不是闲的蛋疼吗？那我为啥不直接用原始的数值变量？

答案是当然可以，而且很多情况下并不会比WOE效果差，因为连续变量的WOE编码确实是损失了一部分信息。但 binning+WOE 能解决一个问题，就是可以把基于对数几率的非线性非单调的特征转化为线性。那这个有什么用呢？在风控场景里，为了保证模型的可解释性，生产中多数还是使用逻辑回归模型进行建模，逻辑回归，也称为对数几率回归，公式如下：
$$
\begin {equation}
\ln(odds) = w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdot \cdot \cdot
\label {eq:4.1}
\end {equation}
$$

式中，$x_i$ 表示样本的第 `i` 个特征，$w_i$ 表示样本的第 `i` 个特征对应的权重，`ln(odds)` 表示对数几率，可以表示为  $ln(odds) = \ln(\frac {p}{1-p})$ ，`p` 是正样本概率。等等，这个 `ln(odds)` 的公式看起来是不是很熟悉？是的，非常像WOE的计算公式，在WOE计算公式里，右半部分 $-\ln(\frac {Bad_{total}}{Good_{total}})$ 对于整个样本集来说是个常数项，左半部分 $\ln(\frac {Bad_{i}}{Good_{i}})$ 其实是对数几率的频率表示，这样转换过后WOE就和逻辑回归的对数几率形成了严格的线性关系，从特征处理的角度也深度吻合了逻辑回归的基本思想，对逻辑回归模型来说是非常必要的。


例如在风控场景里，我们可能用到客户的年龄做特征。我们知道肯定不是年龄越大风险越高，或者年龄越大风险越低，一定是有个年龄段的风险是比其他年龄段高些。还是这个德国评分卡数据集，我们简单画一下年龄和对数几率的关系图

```python
age_odds = np.log(df.groupby('age')['target'].sum() / (df.groupby('age')['target'].count() - df.groupby('age')['target'].sum()))

fig, ax = plt.subplots(figsize=(8,6))
plt.plot(age_odds)
ax.set_title("Odds Distribution by Age", fontsize=18)
ax.set_xlabel("age", fontsize=15)
ax.set_ylabel("Odds", fontsize=15)
```

<img src="https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qX9VP6OS8rMblicLPSe78f1ST6CgmKI0Zib8Q7qKIcbyItP986C1a5yha3ZxGg7rNOG4VOKWQKzd9g/0?wx_fmt=png" style="zoom: 50%;" />

可以看到实际数据年龄和对数几率并不是线性相关的。

为了解决这个非线性相关的问题，我们可以对年龄这个特征做 binning+WOE 变换，用WOE代替年龄段，根据WOE的计算过程，我们可以知道WOE值越高的分类，对数几率越高，违约率也越高，这样就可以把非线性映射转换成线性映射。下图已经把年龄换成了WOE，可以看到WOE和对数几率是线性。

```python
woe_bad_good["woe_odds"] = np.log(woe_bad_good["target_bad_cnt"] / woe_bad_good["target_good_cnt"])

ax = sns.lineplot(x="woe", y="woe_odds", data=woe_bad_good)
ax.set_title("Odds Distribution by Age", fontsize=18)
ax.set_xlabel("age_woe", fontsize=15)
ax.set_ylabel("Odds", fontsize=15)
```

<img src="https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qX9VP6OS8rMblicLPSe78f1wASjBMhr6zI1HOd95nk11jPicw2ct5P1zQxHCxJJs5QnK0u7wYtejdg/0?wx_fmt=png" style="zoom: 50%;" />

**总结一下：WOE可以把相对于Log Odds显现非线性的特征转换为线性的，这对于广义线性模型（如逻辑回归）来说非常有必要。**

## 5 小结

- WOE越大，bad rate越高，也就是说，通过WOE变换，特征值不仅仅代表一个分类，还代表了这个分类的权重。
- WOE可以把相对于Log Odds显现非线性的特征转换为线性的，更符合逻辑回归模型的原理。
- WOE编码对缺失值不敏感，可以单独分一个NA箱。
- WOE对波动不敏感，遇到异常值亦能平稳表现。例如有个人年龄为20，不小心按键盘时按成了200，也不会产生10倍的波动，当然，这个是数据分箱带来的天然优势。


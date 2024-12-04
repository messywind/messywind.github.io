# CS231n


学了那么多理论了做一下斯坦福大学的 CS 课程：CS231n，对 CV 有一个基本的认识，同时加强一下实操能力。

[课程网址](https://cs231n.github.io/)

# 本地环境部署

由于 2024 版的没有给 jupyter 的压缩包，所以先下载 [2024 colab 版本](https://cs231n.github.io/assignments/2024/assignment1_colab.zip)，然后在 colab 上把数据拉下来然后下载到本地，放到 `/datasets` 下，之后删除掉一开始的 `google.colab` 驱动相关。

虚拟环境的话就用 conda 创一个 python3.7 然后使用 [2020 jupyter 版本](https://cs231n.github.io/assignments/2020/assignment1_jupyter.zip)的 `requirements.txt` ，如果有漏包情况再说。

[个人练习 GitHub 地址](https://github.com/messywind/CS231n)

## Assignment 1

### Q1: k-Nearest Neighbor classifier

数据集是一个很多 32 * 32 * 3 (32 × 32 像素，RGB) 的图片分类。

为了方便处理直接压成一行，$32 \times 32 \times 3 = 3072$ 列。

取前 $5000$ 个作为训练集，前 $500$ 个作为测试集。

```python
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print(&#39;Training data shape: &#39;, X_train.shape)
print(&#39;Training labels shape: &#39;, y_train.shape)
print(&#39;Test data shape: &#39;, X_test.shape)
print(&#39;Test labels shape: &#39;, y_test.shape)

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
```

导入 KNN 分类器，他已经把类写好了。

```Python
from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
```

接下来要计算距离。需要写一下 compute_distances_two_loops 的 TODO

```Python
# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)
```
#### TODO: 两重循环计算 L2 距离。

`np.sqrt()` 开根，`np.sum()` 求和。

```Python
    def compute_distances_two_loops(self, X):
        &#34;&#34;&#34;
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        &#34;&#34;&#34;
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```
看一下距离的网格图。

```Python
# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation=&#39;none&#39;)
plt.show()
```

![](/image/ML/CS231n/1.png)

#### Inline Question 1

 Notice the structured patterns in the distance matrix, where some rows or columns are visibly brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)
 
 - What in the data is the cause behind the distinctly bright rows?
 - What causes the columns?
 
 $\color{blue}{\textit Your Answer:}$ *fill this in.*

注意距离矩阵中的结构化模式，其中一些行或列明显更亮。（注意在默认的颜色方案中，黑色表示低距离，而白色表示高距离。）
- 数据中是什么原因导致了这些明显更亮的行？
- 是什么导致了这些明显的列？

$\color{blue}{\textit Your Answer:}$ 行是测试数据，列是训练数据。白色的行是该测试数据远离训练数据，白色的列是该训练数据远离测试数据。

#### TODO: predict_labels

`np.argsort()` 表示返回排序后的原数组下标。这里 `[0 : k]` 取前 $k$ 大，然后再映射到 y

`np.bincount()` 表示将输入数据装进桶计数。

`np.argmax()` 表示取最大值的下标。

```Python
    def predict_labels(self, dists, k=1):
        &#34;&#34;&#34;
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        &#34;&#34;&#34;
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            closest_index = np.argsort(dists[i])[0 : k]
            closest_y = self.y_train[closest_index]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
```

开始 $k = 1$ 的预测。

```Python
# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(&#39;Got %d / %d correct =&gt; accuracy: %f&#39; % (num_correct, num_test, accuracy))
```

`Got 137 / 500 correct =&gt; accuracy: 0.274000`

$k = 5$ 的预测。

```Python
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(&#39;Got %d / %d correct =&gt; accuracy: %f&#39; % (num_correct, num_test, accuracy))
```
`Got 139 / 500 correct =&gt; accuracy: 0.278000`

#### Inline Question 2

We can also use other distance metrics such as L1 distance.
For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$, 

the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
And the pixel-wise mean $\mu_{ij}$ across all images is 
$$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$
The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.

Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply. To clarify, both training and test examples are preprocessed in the same way.

1. Subtracting the mean $\mu$ ($\tilde{p}\_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}\_{ij}^{(k)}=p\_{ij}^{(k)}-\mu_{ij}$.)
3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
5. Rotating the coordinate axes of the data, which means rotating all the images by the same angle. Empty regions in the image caused by rotation are padded with a same pixel value and no interpolation is performed.

$\color{blue}{\textit Your Answer:}$


$\color{blue}{\textit Your Explanation:}$



我们也可以使用其他距离度量方法，比如 L1 距离。
对于某个图像 $I_k$ 中位置 $(i,j)$ 的像素值 $p_{ij}^{(k)}$，

所有图像中所有像素的均值 $\mu$ 为：
$$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$

所有图像中每个像素位置的均值 $\mu_{ij}$ 为：
$$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$

总体标准差 $\sigma$ 和每个像素位置的标准差 $\sigma_{ij}$ 的定义类似。

以下哪种预处理步骤不会改变使用 L1 距离的最近邻分类器的性能？选择所有适用的选项。为明确起见，训练和测试样本都以相同的方式进行预处理。

1. 减去均值 $\mu$ ($\tilde{p}\_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
2. 减去每个像素的均值 $\mu_{ij}$  ($\tilde{p}\_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
3. 减去均值 $\mu$ 并除以标准差 $\sigma$。
4. 减去每个像素的均值 $\mu_{ij}$ 并除以每个像素的标准差 $\sigma_{ij}$。
5. 旋转数据的坐标轴，这意味着将所有图像旋转相同的角度。旋转导致的图像空白区域用相同的像素值填充，不进行插值。

$\color{blue}{\textit Your Answer:}$ 除了 4 都不影响。


$\color{blue}{\textit Your Explanation:}$

#### TODO: 一个循环求 L2 距离：

`np.sum(..., axis=1)` 表示在第一维求和，例如

```Python
import numpy as np

a = np.array([[1, 1, 1], [2, 2, 2]])

print(np.sum(a ** 2, axis=1))
```
得到的结果是 `[3 12]`

这里由于 Numpy 的广播机制，`self.X_train` 的每一行会减掉 `X[i, :]`

例如：

```Python
import numpy as np

a = np.array([[1, 1, 1], [2, 2, 2]])
b = np.array([1, 1, 1])

print(a - b)
```

得到的结果是 `[[0 0 0] [1 1 1]]`

```Python
def compute_distances_one_loop(self, X):
        &#34;&#34;&#34;
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        &#34;&#34;&#34;
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dists[i, :] = np.sqrt(np.sum((self.X_train - X[i, :]) ** 2, axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```

正确性检测：

```Python
# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven&#39;t seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord=&#39;fro&#39;)
print(&#39;One loop difference was: %f&#39; % (difference, ))
if difference &lt; 0.001:
    print(&#39;Good! The distance matrices are the same&#39;)
else:
    print(&#39;Uh-oh! The distance matrices are different&#39;)
```
`One loop difference was: 0.000000
Good! The distance matrices are the same`

#### TODO: 不使用循环求 L2 距离

展开平方。$(x_i - x_j)^ 2 = x_i ^ 2 &#43; x_j ^ 2 - 2x_ix_j$

测试集这里要用 `.reshape((num_test, 1))`，利用广播机制把 `test_squared &#43; train_squared` 弄成 `(num_test, num_train)` 大小，然后注意矩阵乘法的时候要满足中间的矩阵大小是相等的，所以训练集要转置一下，即 `self.X_train.T`

```Python
    def compute_distances_no_loops(self, X):
        &#34;&#34;&#34;
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        &#34;&#34;&#34;
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        test_squared = np.sum(X ** 2, axis=1).reshape((num_test, 1))
        
        train_squared = np.sum(self.X_train ** 2, axis=1)
        
        cross_term = np.dot(X, self.X_train.T)
        
        dists = np.sqrt(test_squared &#43; train_squared - 2 * cross_term)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```
正确性检测：

```Python
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord=&#39;fro&#39;)
print(&#39;No loop difference was: %f&#39; % (difference, ))
if difference &lt; 0.001:
    print(&#39;Good! The distance matrices are the same&#39;)
else:
    print(&#39;Uh-oh! The distance matrices are different&#39;)
```
`No loop difference was: 0.000000
Good! The distance matrices are the same`

时间对比：
```Python
# Let&#39;s compare how fast the implementations are
def time_function(f, *args):
    &#34;&#34;&#34;
    Call a function f with args and return the time (in seconds) that it took to execute.
    &#34;&#34;&#34;
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print(&#39;Two loop version took %f seconds&#39; % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print(&#39;One loop version took %f seconds&#39; % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print(&#39;No loop version took %f seconds&#39; % no_loop_time)

# You should see significantly faster performance with the fully vectorized implementation!

# NOTE: depending on what machine you&#39;re using, 
# you might not see a speedup when you go from two loops to one loop, 
# and might even see a slow-down.
```

```{title=&#34;Output&#34;}
Two loop version took 27.218139 seconds
One loop version took 40.968490 seconds
No loop version took 0.350827 seconds
```

交叉验证 (Cross-validation)：

#### TODO: 分成 folds

大概就是把数据集切分成若干个部分，然后遍历 folds，把当前作为测试集其他作为训练集，再逐个枚举 $k$，得到一个平均值。

```Python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for k in k_choices:
    k_to_accuracies[k] = []
    for fold in range(num_folds):
        # 使用除当前 fold 以外的所有 fold 作为训练数据
        X_train_fold = np.concatenate([X_train_folds[i] for i in range(num_folds) if i != fold])
        y_train_fold = np.concatenate([y_train_folds[i] for i in range(num_folds) if i != fold])
        
        # 当前 fold 作为验证集
        X_val_fold = X_train_folds[fold]
        y_val_fold = y_train_folds[fold]

        # 训练 k-NN 分类器
        classifier = KNearestNeighbor()
        classifier.train(X_train_fold, y_train_fold)
        
        # 在验证集上进行预测
        y_val_pred = classifier.predict(X_val_fold, k=k, num_loops=0)
        
        # 计算准确率
        num_correct = np.sum(y_val_pred == y_val_fold)
        accuracy = num_correct / len(y_val_fold)
        k_to_accuracies[k].append(accuracy)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print(&#39;k = %d, accuracy = %f&#39; % (k, accuracy))
```

```{title=&#34;Output&#34;}
k = 1, accuracy = 0.263000
k = 1, accuracy = 0.257000
k = 1, accuracy = 0.264000
k = 1, accuracy = 0.278000
k = 1, accuracy = 0.266000
k = 3, accuracy = 0.239000
k = 3, accuracy = 0.249000
k = 3, accuracy = 0.240000
k = 3, accuracy = 0.266000
k = 3, accuracy = 0.254000
k = 5, accuracy = 0.248000
k = 5, accuracy = 0.266000
k = 5, accuracy = 0.280000
k = 5, accuracy = 0.292000
k = 5, accuracy = 0.280000
k = 8, accuracy = 0.262000
k = 8, accuracy = 0.282000
k = 8, accuracy = 0.273000
k = 8, accuracy = 0.290000
k = 8, accuracy = 0.273000
k = 10, accuracy = 0.265000
k = 10, accuracy = 0.296000
k = 10, accuracy = 0.276000
k = 10, accuracy = 0.284000
k = 10, accuracy = 0.280000
k = 12, accuracy = 0.260000
k = 12, accuracy = 0.295000
k = 12, accuracy = 0.279000
k = 12, accuracy = 0.283000
k = 12, accuracy = 0.280000
k = 15, accuracy = 0.252000
k = 15, accuracy = 0.289000
k = 15, accuracy = 0.278000
k = 15, accuracy = 0.282000
k = 15, accuracy = 0.274000
k = 20, accuracy = 0.270000
k = 20, accuracy = 0.279000
k = 20, accuracy = 0.279000
k = 20, accuracy = 0.282000
k = 20, accuracy = 0.285000
k = 50, accuracy = 0.271000
k = 50, accuracy = 0.288000
k = 50, accuracy = 0.278000
k = 50, accuracy = 0.269000
k = 50, accuracy = 0.266000
k = 100, accuracy = 0.256000
k = 100, accuracy = 0.270000
k = 100, accuracy = 0.263000
k = 100, accuracy = 0.256000
k = 100, accuracy = 0.263000
```

画出不同 k 的准确率图。

```Python
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title(&#39;Cross-validation on k&#39;)
plt.xlabel(&#39;k&#39;)
plt.xticks(np.arange(min(k_choices), max(k_choices) &#43; 1, 4))
plt.ylabel(&#39;Cross-validation accuracy&#39;)
plt.show()
```
![](/image/ML/CS231n/2.png)

选一个最佳 $k$ 来预测数据，这里选择 $k = 10$，要求准确率应该在 28% 以上。

```Python
# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(&#39;Got %d / %d correct =&gt; accuracy: %f&#39; % (num_correct, num_test, accuracy))
```

`Got 141 / 500 correct =&gt; accuracy: 0.282000` 预测准确率为 28.2%

#### Inline Question 3

Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.
1. The decision boundary of the k-NN classifier is linear.
2. The training error of a 1-NN will always be lower than or equal to that of 5-NN.
3. The test error of a 1-NN will always be lower than that of a 5-NN.
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
5. None of the above.

$\color{blue}{\textit Your Answer:}$


$\color{blue}{\textit Your Explanation:}$

**内嵌问题 3**

关于 $k$-最近邻（$k$-NN）在分类设置中的以下哪些陈述是正确的，并且适用于所有 $k$？选择所有适用的选项。
1. $k$-NN 分类器的决策边界是线性的。
2. 1-NN 的训练误差总是小于或等于 5-NN 的训练误差。
3. 1-NN 的测试误差总是小于 5-NN 的测试误差。
4. 使用 $k$-NN 分类器对测试样本进行分类所需的时间随着训练集的大小而增加。
5. 以上都不正确。

$\color{blue}{\textit Your Answer:}$ 4


$\color{blue}{\textit Your Explanation:}$ 1 显然不对，2、3 直接看结果，4 确实是因为距离是要遍历数据集。

## 参考
https://github.com/Divsigma/2020-cs213n/tree/master/cs231n

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlcs231n/  


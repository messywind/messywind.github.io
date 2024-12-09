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

### Q2: Training a Support Vector Machine

还是那个数据集，这次先减去一个图像像素的平均值。然后因为是 SVM，所以 `np.ones` 加上一个 $1$ 的偏置。

![1](/image/ML/CS231n/wb.jpeg)

```Python
# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype(&#39;uint8&#39;)) # visualize the mean image
plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
```
#### TODO: svm_loss_naive

计算损失的时候，用 `X[i].dot(W)` 表示该条数据在 $10$ 个分类下的表现得分 (W 是 3073 * 10 的矩阵)，`correct_class_score = scores[y[i]]` 表示从得分向量中提取出第 $i$ 个样本的正确类别的得分。根据官网讲义(https://cs231n.github.io/linear-classify/)，定义损失为 $L_i = \sum\limits_{j \ne y_i}\max(0, s_j - s_{y_i} &#43; \Delta)$
为什么这么定义呢？根据讲义
&gt; The Multiclass Support Vector Machine &#34;wants&#34; the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.

多类别支持向量机 &#34;希望 &#34;正确类别的得分至少比所有其他类别的得分高出 delta 值。 如果任何一个类别的得分在红色区域内（或更高），那么就会有累计损失。 否则，损失为零。 我们的目标是为训练数据中的所有示例找到同时满足这一约束条件的权重，并尽可能降低总损失。

![1](/image/ML/CS231n/margin.jpg)

除此之外，还要加上一个正则化损失，一般是 L2：

$$
R(W) = \sum_{k}\sum_{l}W_{k, l} ^ 2
$$

然后乘以一个参数(函数传入的 reg)，最后和平均累计损失相加得到最终损失：

$$
L = \frac{1}{N}\sum_i L_i &#43; \lambda R(W)
$$

那么梯度就是对损失函数对 W 求导，第一部分可以在算损失的时候计算出来。

```Python
def svm_loss_naive(W, X, y, reg):
    &#34;&#34;&#34;
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 &lt;= c &lt; C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    &#34;&#34;&#34;
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score &#43; 1  # note delta = 1
            if margin &gt; 0:
                loss &#43;= margin
                dW[:, j] &#43;= X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss &#43;= reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW &#43;= 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```

检验一下梯度是否对

```Python
# Once you&#39;ve implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from cs231n.gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn&#39;t forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)
```

```{title=&#34;Output&#34;}
numerical: 10.450954 analytic: 10.450954, relative error: 2.255776e-11
numerical: 7.036361 analytic: 7.125610, relative error: 6.301992e-03
numerical: -33.423497 analytic: -33.423497, relative error: 9.550081e-12
numerical: 22.845277 analytic: 22.830022, relative error: 3.340013e-04
numerical: 11.480153 analytic: 11.564991, relative error: 3.681392e-03
numerical: -26.401444 analytic: -26.401444, relative error: 1.537030e-11
numerical: 8.954193 analytic: 8.954193, relative error: 2.288146e-11
numerical: -3.249659 analytic: -3.249659, relative error: 7.932744e-12
numerical: 9.067911 analytic: 9.075238, relative error: 4.038672e-04
numerical: -12.493792 analytic: -12.493792, relative error: 1.033714e-11
numerical: -14.040295 analytic: -14.040295, relative error: 2.451599e-11
numerical: -17.850408 analytic: -17.850408, relative error: 4.971397e-12
numerical: 4.775439 analytic: 4.811624, relative error: 3.774404e-03
numerical: 1.002399 analytic: 1.002399, relative error: 1.362218e-10
numerical: -19.384504 analytic: -19.382918, relative error: 4.091049e-05
numerical: 23.624824 analytic: 23.624824, relative error: 1.588455e-11
numerical: -26.578911 analytic: -26.578911, relative error: 3.778756e-12
numerical: -9.700067 analytic: -9.700067, relative error: 3.511119e-12
numerical: 2.004068 analytic: 2.004068, relative error: 1.303094e-11
numerical: -12.635156 analytic: -12.635156, relative error: 1.036973e-11
```
可以看到相对误差都比较小。

#### Inline Question 1

It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? How would change the margin affect of the frequency of this happening? *Hint: the SVM loss function is not strictly speaking differentiable*

$\color{blue}{\textit Your Answer:}$ *fill this in.*  




有时候，梯度检查中的某个维度可能不会完全匹配。这种差异可能是由什么引起的？这是否是一个值得担心的问题？在一维中，梯度检查可能失败的一个简单例子是什么？改变边距会如何影响这种情况发生的频率？*提示：SVM损失函数严格来说并不是可微的*

$\color{blue}{\textit Your Answer:}$ 在梯度检查中，某个维度不完全匹配的差异可能是由于数值计算的精度限制或损失函数的不可微性引起的。SVM损失函数在某些点上是不可微的，例如在边界条件下（即损失函数的“铰链”部分），这可能导致梯度检查不精确。

这种差异通常不是一个严重的问题，因为数值梯度计算本身就有一定的误差。一个简单的例子是在一维中，考虑一个绝对值函数 $( f(x) = |x| )$，在 $( x = 0 )$ 处，梯度是不可定义的，这可能导致梯度检查失败。

改变边距（margin）可能会影响这种情况发生的频率。较大的边距可能会减少不可微点的数量，从而减少梯度检查失败的可能性。然而，边距的改变也会影响模型的性能，因此需要在准确性和稳定性之间进行权衡。


#### TODO: svm_loss_vectorized
损失计算：

`scores = X.dot(W)`，得到一个 $N \times 10$ 的每个样本每个分类评分数组。

`correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)` 将每个样本的正确分类评分拿出来，并转成 $N \times 1$ 的列向量方便广播。

`margins = np.maximum(0, scores - correct_class_scores &#43; 1)` `margins[np.arange(num_train), y] = 0` 将每个分数减去正确分数，然后把正确分数的那一列变成 $0$

梯度计算：

`binary = margins &gt; 0` `binary = binary.astype(float)` 得到一个每个元素是否大于 $0$ 的矩阵并转浮点数。

`row_sum = np.sum(binary, axis=1)` ，每一行有多少元素大于 $0$

`binary[np.arange(num_train), y] = -row_sum`，将每个正确位置的地方置为负的 `row_sum`，因为大于零的都会产生负贡献。

`dW = X.T.dot(binary)`，相当于给每个特征都做 naive 版本的这个操作：`dW[:, j] &#43;= X[i]` `dW[:, y[i]] -= X[i]`

```Python
def svm_loss_vectorized(W, X, y, reg):
    &#34;&#34;&#34;
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    &#34;&#34;&#34;
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores &#43; 1)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins) / num_train
    loss &#43;= reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    binary = margins &gt; 0
    binary = binary.astype(float)
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum
    dW = X.T.dot(binary) / num_train
    dW &#43;= 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```

对比一下

```Python
# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print(&#39;Naive loss: %e computed in %fs&#39; % (loss_naive, toc - tic))

from cs231n.classifiers.linear_svm import svm_loss_vectorized
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print(&#39;Vectorized loss: %e computed in %fs&#39; % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be much faster.
print(&#39;difference: %f&#39; % (loss_naive - loss_vectorized))
```

```{title=&#34;Output&#34;}
Naive loss: 9.399452e&#43;00 computed in 0.112582s
Vectorized loss: 9.399452e&#43;00 computed in 0.008816s
difference: -0.000000
```

```Python
# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print(&#39;Naive loss and gradient: computed in %fs&#39; % (toc - tic))

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print(&#39;Vectorized loss and gradient: computed in %fs&#39; % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord=&#39;fro&#39;)
print(&#39;difference: %f&#39; % difference)
```

```{title=&#34;Output&#34;}
Naive loss and gradient: computed in 0.105182s
Vectorized loss and gradient: computed in 0.002997s
difference: 0.000000
```

#### TODO: LinearClassifier

每次随机抽 `batch_size` 个样本计算梯度，迭代更新 W

```Python
#########################################################################
# TODO:                                                                 #
# Sample batch_size elements from the training data and their           #
# corresponding labels to use in this round of gradient descent.        #
# Store the data in X_batch and their corresponding labels in           #
# y_batch; after sampling X_batch should have shape (batch_size, dim)   #
# and y_batch should have shape (batch_size,)                           #
#                                                                       #
# Hint: Use np.random.choice to generate indices. Sampling with         #
# replacement is faster than sampling without replacement.              #
#########################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

indices = np.random.choice(num_train, batch_size, replace=True)
X_batch = X[indices]
y_batch = y[indices]

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# evaluate loss and gradient
loss, grad = self.loss(X_batch, y_batch, reg)
loss_history.append(loss)

# perform parameter update
#########################################################################
# TODO:                                                                 #
# Update the weights using the gradient and the learning rate.          #
#########################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

self.W -= learning_rate * grad

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

#### TODO: predict

直接用数据乘一下已经训练好了的 W 矩阵，argmax 取出最高分数的类。

```Python
    def predict(self, X):
        &#34;&#34;&#34;
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        &#34;&#34;&#34;
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred
```

```{title=&#34;Output&#34;}
iteration 0 / 1500: loss 791.498502
iteration 100 / 1500: loss 289.450806
iteration 200 / 1500: loss 107.822187
iteration 300 / 1500: loss 42.558331
iteration 400 / 1500: loss 18.847986
iteration 500 / 1500: loss 10.427407
iteration 600 / 1500: loss 6.668877
iteration 700 / 1500: loss 5.704408
iteration 800 / 1500: loss 5.485960
iteration 900 / 1500: loss 5.553929
iteration 1000 / 1500: loss 6.068083
iteration 1100 / 1500: loss 5.816354
iteration 1200 / 1500: loss 5.403992
iteration 1300 / 1500: loss 5.660007
iteration 1400 / 1500: loss 5.307236
That took 7.161625s
```

![1](/image/ML/CS231n/3.png)

#### TODO: 使用不同学习率和正则化参数

```Python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.39 (&gt; 0.385) on the validation set.

# Note: you may see runtime/overflow warnings during hyper-parameter search. 
# This may be caused by extreme values, and is not a bug.

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don&#39;t take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
    
        svm.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1500, verbose=False)
        
        # 计算训练集上的准确率
        y_train_pred = svm.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        
        # 计算验证集上的准确率
        y_val_pred = svm.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        
        # 将结果存储在字典中
        results[(lr, reg)] = (train_accuracy, val_accuracy)
        
        # 如果当前验证准确率是最高的，则更新 best_val 和 best_svm
        if val_accuracy &gt; best_val:
            best_val = val_accuracy
            best_svm = svm

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print(&#39;lr %e reg %e train accuracy: %f val accuracy: %f&#39; % (
                lr, reg, train_accuracy, val_accuracy))
    
print(&#39;best validation accuracy achieved during cross-validation: %f&#39; % best_val)
```

```{title=&#34;Output&#34;}
lr 1.000000e-07 reg 2.500000e&#43;04 train accuracy: 0.369000 val accuracy: 0.379000
lr 1.000000e-07 reg 5.000000e&#43;04 train accuracy: 0.349898 val accuracy: 0.364000
lr 5.000000e-05 reg 2.500000e&#43;04 train accuracy: 0.075143 val accuracy: 0.094000
lr 5.000000e-05 reg 5.000000e&#43;04 train accuracy: 0.100265 val accuracy: 0.087000
best validation accuracy achieved during cross-validation: 0.379000
```
可视化权重：

![1](/image/ML/CS231n/4.png)

#### Inline question 2

Describe what your visualized SVM weights look like, and offer a brief explanation for why they look the way they do.

$\color{blue}{\textit Your Answer:}$ *fill this in*

当然，这里是翻译：

**内联问题 2**

描述你可视化的 SVM 权重是什么样的，并简要解释它们为什么会是这样的。

$\color{blue}{\textit Your Answer:}$ 

1. **权重图像的外观**：
   - 每个类别的权重图像可能看起来像该类别的典型代表。例如，飞机类别的权重图像可能会显示出机翼的形状，汽车类别可能会显示出车轮的形状。
   - 这些图像通常是模糊的，因为 SVM 是线性分类器，它试图通过线性组合输入特征来区分类别。

2. **为什么权重看起来是这样的**：
   - SVM 权重反映了模型在训练过程中学到的特征，这些特征有助于区分不同的类别。
   - 权重的正值区域表示该区域的像素对该类别的正贡献，而负值区域表示对该类别的负贡献。
   - 由于 SVM 是线性模型，它只能捕捉到线性可分的特征，因此权重图像可能无法捕捉到复杂的非线性特征。

3. **权重的可解释性**：
   - 通过观察这些权重图像，可以直观地理解模型在做出分类决策时关注的图像区域。
   - 这有助于调试和改进模型，例如通过数据增强或特征提取来提高模型的性能。


### Q3: Implement a Softmax classifier

![1](/image/ML/CS231n/softmax.webp)

#### SoftMax 损失函数

SoftMax 是把得分转换成了概率。公式如下：

$$
S(y_i) = \dfrac{e ^ {y_i}}{\sum\limits_{j} e ^ {y_j}}
$$

损失函数就是根据交叉熵套了个 $-\log(x)$：

$$
L_i = -\log\left(\dfrac{e ^ {y_i}}{\sum\limits_{j} e ^ {y_j}}\right)
$$

#### SoftMax 梯度推导

首先样本 $i$ 的得分为：

$$
s_i = x_i \cdot W
$$

$s_{i, j}$ 表示样本 $i$ 在类别 $j$ 上的得分。

$$
p_i = \text{softmax}(s_i) = \dfrac{e ^ {s_i}}{\sum\limits_{k} e ^ {s_{i, k}}}
$$

$p_{i, j}$ 表示样本 $i$ 被预测为类别 $j$ 的概率。

假设一共有 $C$ 类，$p_i$ 是长这样子的：

$$
p_i = \left[ \frac{e^{s_{i, 1}}}{\sum\limits_{k=1}^{C} e^{s_{i, k}}}, \frac{e^{s_{i, 2}}}{\sum\limits_{k=1}^{C} e^{s_{i, k}}}, \cdots, \frac{e^{s_{i, C}}}{\sum\limits_{k=1}^{C} e^{s_{i, k}}} \right]
$$

那么它的损失函数为：

$$
L_i = -\log(p_i) = -\log\left(\dfrac{e ^ {s_i}}{\sum\limits_{k} e ^ {s_{i, k}}}\right)
$$

(以下公式为了形式美观将 $s_{i, j}$ 令成 $s_j$，$p_{i, j}$ 令成 $p_j$，意思是都是样本 $i$ 的)

损失函数对 $W$ 求导，并使用链式法则：

$$
\dfrac{\partial L_i}{\partial W} = \dfrac{\partial L_i}{\partial s_j} \times \dfrac{\partial s_j}{\partial W}
$$

显然有 $\dfrac{\partial s_j}{\partial W} = x_i$，重点讨论 $\dfrac{\partial L_i}{\partial s_j}$：

{{&lt; admonition type=&#34;tip&#34; title=&#34;$\frac{\partial L_i}{\partial s_j}$&#34;&gt;}}

对于每个类别 $j$：

- 如果 $j$ 为正确类别 ($y_i = j$)：

$$
\frac{\partial L_i}{\partial s_j} = \frac{\partial (-\log(p_j))}{\partial s_j} = -\frac{1}{p_j} \times \frac{\partial p_j}{\partial s_j}
$$

接下来 $\dfrac{\partial p_j}{\partial s_j}$ 是：

$$
\frac{\partial p_j}{\partial s_j} = \frac{\partial}{\partial s_j} \left( \frac{e^{s_j}}{\sum\limits_{k} e^{s_{j, k}}} \right)
$$

求导：

$$
\frac{\partial}{\partial s_j} \left( \frac{e^{s_j}}{\sum\limits_{k} e^{s_{j, k}}} \right) = \frac{e^{s_j} \sum\limits_{k} e^{s_{j, k}} - e^{s_j} \cdot e^{s_j}}{\left(\sum\limits_{k} e^{s_{j, k}}\right)^2} = \frac{e^{s_j} \left(\sum\limits_{k} e^{s_{j, k}} - e^{s_j}\right)}{\left(\sum\limits_{k} e^{s_{j, k}}\right)^2} = \dfrac{e ^ {s_j}}{\sum\limits_{k} e^{s_{j, k}}} \times \left(1 - \dfrac{e ^ {s_j}}{\sum\limits_{k} e^{s_{j, k}}}\right) = p_j (1 - p_j)
$$

于是 $\dfrac{\partial L_i}{\partial s_j} = -\dfrac{1}{p_j} \times p_j(1 - p_j) = (p_j - 1)$

则 $\dfrac{\partial L_i}{\partial W} = (p_j - 1) x_i$

- 如果 $j$ 为不正确类别 ($y_i \ne j$)：

$$
\frac{\partial L_i}{\partial s_{y_j}} = \frac{\partial (-\log(p_{y_i}))}{\partial s_{y_j}} = -\frac{1}{p_{y_i}} \times \frac{\partial p_{y_i}}{\partial s_{y_j}}
$$

接下来 $\dfrac{\partial p_{y_j}}{\partial s_{y_j}}$ 是：

$$
\dfrac{\partial p_{y_j}}{\partial s_{y_j}} = \frac{\partial}{\partial s_{y_j}} \left( \frac{e^{s_j}}{\sum\limits_{k} e^{s_{j, k}}} \right)
$$

求导：

$$
\frac{\partial}{\partial s_{y_j}} \left( \frac{e^{s_j}}{\sum\limits_{k} e^{s_{j, k}}} \right) = -\frac{e^{s_j} e ^ {s_{y_j}}}{\left(\sum\limits_{k} e^{s_{j, k}}\right) ^ 2} = -p_j p_{y_j}
$$

于是 $\dfrac{\partial L_i}{\partial s_{y_j}} = -\dfrac{1}{p_{y_i}} \times \dfrac{\partial p_{y_i}}{\partial s_{y_j}} = p_j$

则 $\dfrac{\partial L_i}{\partial W} = p_j x_i$

{{&lt; /admonition &gt;}}

#### TODO: softmax_loss_naive

```Python
def softmax_loss_naive(W, X, y, reg):
    &#34;&#34;&#34;
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 &lt;= c &lt; C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    &#34;&#34;&#34;
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don&#39;t forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取样本数量和类别数量
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    # 遍历每个样本
    for i in range(num_train):
        # 计算得分
        scores = X[i].dot(W)
        
        # 数值稳定性处理
        scores -= np.max(scores)
        
        # 计算softmax概率
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        # 计算损失
        loss &#43;= -np.log(probs[y[i]])
        
        # 计算梯度
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] &#43;= (probs[j] - 1) * X[i]
            else:
                dW[:, j] &#43;= probs[j] * X[i]
    
    # 平均损失
    loss /= num_train
    # 加上正则化损失
    loss &#43;= 0.5 * reg * np.sum(W * W)
    
    # 平均梯度
    dW /= num_train
    # 加上正则化梯度
    dW &#43;= reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```
#### Inline Question 1

Why do we expect our loss to be close to -log(0.1)? Explain briefly.**

$\color{blue}{\textit Your Answer:}$ *Fill this in* 


**内嵌问题 1**

为什么我们期望损失接近于 $-\log(0.1)$？请简要解释。

$\color{blue}{\textit Your Answer:}$ 

在 Softmax 分类器中，损失函数的计算是基于预测概率的对数损失。假设我们有 10 个类别，并且权重矩阵是随机初始化的，那么每个类别的预测概率大约是均匀分布的，即每个类别的概率约为 $0.1$

因此，损失函数的期望值为 $-\log(0.1)$，因为这是对数损失在预测概率为 $0.1$ 时的值。

#### TODO: softmax_loss_vectorized

```Python
def softmax_loss_vectorized(W, X, y, reg):
    &#34;&#34;&#34;
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    &#34;&#34;&#34;
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don&#39;t forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取样本数量
    num_train = X.shape[0]

    # 计算得分矩阵
    scores = X.dot(W)
    
    # 数值稳定性处理
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 计算softmax概率
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 计算损失
    correct_log_probs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_log_probs) / num_train
    loss &#43;= 0.5 * reg * np.sum(W * W)
    
    # 计算梯度
    dscores = probs
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    dW &#43;= reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```

对比：

```Python
# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print(&#39;naive loss: %e computed in %fs&#39; % (loss_naive, toc - tic))

from cs231n.classifiers.softmax import softmax_loss_vectorized
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print(&#39;vectorized loss: %e computed in %fs&#39; % (loss_vectorized, toc - tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord=&#39;fro&#39;)
print(&#39;Loss difference: %f&#39; % np.abs(loss_naive - loss_vectorized))
print(&#39;Gradient difference: %f&#39; % grad_difference)
```

```{title=&#34;Output&#34;}
naive loss: 2.304545e&#43;00 computed in 0.112797s
vectorized loss: 2.304545e&#43;00 computed in 0.007444s
Loss difference: 0.000000
Gradient difference: 0.000000
```

#### TODO: 交叉验证

```Python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.

from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
for lr in learning_rates:
    for reg in regularization_strengths:
        softmax = Softmax()
        
        softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1500, verbose=False)
        
        y_train_pred = softmax.predict(X_train)
        y_val_pred = softmax.predict(X_val)
        
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        
        results[(lr, reg)] = (train_accuracy, val_accuracy)
        
        if val_accuracy &gt; best_val:
            best_val = val_accuracy
            best_softmax = softmax

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print(&#39;lr %e reg %e train accuracy: %f val accuracy: %f&#39; % (
                lr, reg, train_accuracy, val_accuracy))
    
print(&#39;best validation accuracy achieved during cross-validation: %f&#39; % best_val)
```

```{title=&#34;Output&#34;}
lr 1.000000e-07 reg 2.500000e&#43;04 train accuracy: 0.345571 val accuracy: 0.366000
lr 1.000000e-07 reg 5.000000e&#43;04 train accuracy: 0.327163 val accuracy: 0.345000
lr 5.000000e-07 reg 2.500000e&#43;04 train accuracy: 0.340449 val accuracy: 0.352000
lr 5.000000e-07 reg 5.000000e&#43;04 train accuracy: 0.329878 val accuracy: 0.330000
best validation accuracy achieved during cross-validation: 0.366000
```

#### Inline Question 2 - *True or False*

Suppose the overall training loss is defined as the sum of the per-datapoint loss over all training examples. It is possible to add a new datapoint to a training set that would leave the SVM loss unchanged, but this is not the case with the Softmax classifier loss.

$\color{blue}{\textit Your Answer:}$


$\color{blue}{\textit Your Explanation:}$

**内嵌问题 2** - *对或错*

假设整体训练损失定义为所有训练样本的每个数据点损失之和。可以添加一个新的数据点到训练集中，使得 SVM 损失保持不变，但对于 Softmax 分类器损失来说，这种情况不会发生。

$\color{blue}{\textit Your Answer:}$ True

$\color{blue}{\textit Your Explanation:}$


在 SVM 中，损失函数是基于边界的。对于一个新的数据点，如果它位于正确的边界一侧并且离边界足够远，那么它对损失的贡献为零，因此不会改变整体损失。

然而，在 Softmax 分类器中，损失函数是基于概率分布的对数损失。每个数据点都会对损失产生影响，因为即使是一个新的数据点也会改变概率分布，从而影响损失。因此，添加一个新的数据点总是会改变 Softmax 分类器的损失。

可视化权重：

![1](/image/ML/CS231n/5.png)

## 参考
https://github.com/Divsigma/2020-cs213n/tree/master/cs231n

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlcs231n/  


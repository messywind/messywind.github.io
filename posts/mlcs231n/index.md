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
 
 ${\textit Your Answer:}$ *fill this in.*

注意距离矩阵中的结构化模式，其中一些行或列明显更亮。（注意在默认的颜色方案中，黑色表示低距离，而白色表示高距离。）
- 数据中是什么原因导致了这些明显更亮的行？
- 是什么导致了这些明显的列？

${\textit Your Answer:}$ 行是测试数据，列是训练数据。白色的行是该测试数据远离训练数据，白色的列是该训练数据远离测试数据。

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

${\textit Your Answer:}$


${\textit Your Explanation:}$



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

${\textit Your Answer:}$ 除了 4 都不影响。


${\textit Your Explanation:}$

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

${\textit Your Answer:}$


${\textit Your Explanation:}$

**内嵌问题 3**

关于 $k$-最近邻（$k$-NN）在分类设置中的以下哪些陈述是正确的，并且适用于所有 $k$？选择所有适用的选项。
1. $k$-NN 分类器的决策边界是线性的。
2. 1-NN 的训练误差总是小于或等于 5-NN 的训练误差。
3. 1-NN 的测试误差总是小于 5-NN 的测试误差。
4. 使用 $k$-NN 分类器对测试样本进行分类所需的时间随着训练集的大小而增加。
5. 以上都不正确。

${\textit Your Answer:}$ 4


${\textit Your Explanation:}$ 1 显然不对，2、3 直接看结果，4 确实是因为距离是要遍历数据集。

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

${\textit Your Answer:}$ *fill this in.*  




有时候，梯度检查中的某个维度可能不会完全匹配。这种差异可能是由什么引起的？这是否是一个值得担心的问题？在一维中，梯度检查可能失败的一个简单例子是什么？改变边距会如何影响这种情况发生的频率？*提示：SVM损失函数严格来说并不是可微的*

${\textit Your Answer:}$ 在梯度检查中，某个维度不完全匹配的差异可能是由于数值计算的精度限制或损失函数的不可微性引起的。SVM损失函数在某些点上是不可微的，例如在边界条件下（即损失函数的“铰链”部分），这可能导致梯度检查不精确。

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

${\textit Your Answer:}$ *fill this in*

当然，这里是翻译：

**内联问题 2**

描述你可视化的 SVM 权重是什么样的，并简要解释它们为什么会是这样的。

${\textit Your Answer:}$ 

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

Why do we expect our loss to be close to -log(0.1)? Explain briefly.

${\textit Your Answer:}$ *Fill this in* 


**内嵌问题 1**

为什么我们期望损失接近于 $-\log(0.1)$？请简要解释。

${\textit Your Answer:}$ 

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

${\textit Your Answer:}$


${\textit Your Explanation:}$

**内嵌问题 2** - *对或错*

假设整体训练损失定义为所有训练样本的每个数据点损失之和。可以添加一个新的数据点到训练集中，使得 SVM 损失保持不变，但对于 Softmax 分类器损失来说，这种情况不会发生。

${\textit Your Answer:}$ True

${\textit Your Explanation:}$


在 SVM 中，损失函数是基于边界的。对于一个新的数据点，如果它位于正确的边界一侧并且离边界足够远，那么它对损失的贡献为零，因此不会改变整体损失。

然而，在 Softmax 分类器中，损失函数是基于概率分布的对数损失。每个数据点都会对损失产生影响，因为即使是一个新的数据点也会改变概率分布，从而影响损失。因此，添加一个新的数据点总是会改变 Softmax 分类器的损失。

可视化权重：

![1](/image/ML/CS231n/5.png)

### Q4: Two-Layer Neural Network

首先检验一下前向传播，数据生成一个两组大小为 $4 \times 5 \times 6$ 的数据，也就是说输入层是 $120$ 个节点，然后直接连接输出层，设输出层有 $3$ 个节点。
所以说 $w$ 就是 $120 \times 3$ 的，$b$ 就是 $3$ 的。

```Python
# Test the affine_forward function

num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

out, _ = affine_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

# Compare your output with ours. The error should be around e-9 or less.
print(&#39;Testing affine_forward function:&#39;)
print(&#39;difference: &#39;, rel_error(out, correct_out))
```

#### TODO: affine_forward

把 $x$ 压成 $120$ 的第二维。

```Python
def affine_forward(x, w, b):
    &#34;&#34;&#34;
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    &#34;&#34;&#34;
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.reshape(x.shape[0], -1).dot(w) &#43; b
    cache = (x, w, b)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache
```

接下来检验反向传播。

#### TODO: affine_backward

我们知道这是批量梯度下降，也就是说输入数据 $x$ 被我们处理成了 $n \times k$ 维的，每一行代表一个数据，列就是特征。那么输出层定义为 $n \times m$ 维的，$n$ 是数据批量个数，$m$ 是输出层节点个数，但是 $w$ 和 $b$ 分别是 $k \times m$ 与 $1 \times m$ 维的，也就是说这 $n$ 个数据共用 $w$ 和 $b$，然后每个数据对应一组输出。

那么在反向传播时，我们需要计算 $x, w, b$ 的偏导，首先他给了一个 `dout` 数组，这代表输出层的导数，他也是一个 $n \times m$ 的。

单看 $\text{out}_{i, j}$：

$$
\text{out}\_{i, j} = \sum_k x_{i, k} w_{k, j} &#43; b_j
$$

$\text{out}\_{i, j}$ 对 $w_{k, j}$ 求偏导：

$$
\dfrac{\partial \text{out}\_{i, j}}{\partial w_{k, j}} = x_{i, k}
$$

$\text{out}\_{i, j}$ 对 $x_{i, k}$ 求偏导：

$$
\dfrac{\partial \text{out}\_{i, j}}{\partial x_{i, k}} = w_{k, j}
$$

$\text{out}\_{i, j}$ 对 $b_j$ 求偏导：

$$
\dfrac{\partial \text{out}\_{i, j}}{\partial b_j} = 1
$$

那么损失函数 $L$ 对 $w_{k, j}$ 求偏导，根据链式法则有：

$$
\dfrac{\partial L}{\partial w_{k, j}} = \sum_i \dfrac{\partial L}{\partial \text{out}\_{i, j}} \dfrac{\partial \text{out}\_{i, j}}{\partial w_{k, j}}
$$

其中 $\dfrac{\partial L}{\partial \text{out}\_{i, j}}$ 这个东西就是 `dout[i, j]` (上游传来的导数)，而 $\dfrac{\partial \text{out}\_{i, j}}{\partial w_{k, j}} = x_{i, k}$，所以

$$
\dfrac{\partial L}{\partial w_{k, j}} = \sum_i \text{dout}\_{i, j}x_{i, k}
$$

再来看损失函数 $L_i$ 对 $x_{i, k}$ 求偏导，根据链式法则有：

$$
\dfrac{\partial L}{\partial x_{i, k}} = \sum_j \dfrac{\partial L}{\partial \text{out}\_{i, j}} \dfrac{\partial \text{out}\_{i, j}}{\partial x_{i, k}}
$$

继续带入 $\dfrac{\partial L}{\partial \text{out}\_{i, j}} = \text{dout}\_{i, j}$，$\dfrac{\partial \text{out}\_{i, j}}{\partial x_{i, k}} = w_{k, j}$：

$$
\dfrac{\partial L}{\partial x_{i, k}} = \sum_j \text{dout}\_{i, j}w_{k, j}
$$

继续看损失函数 $L_i$ 对 $b_{j}$ 求偏导，根据链式法则有：

$$
\dfrac{\partial L}{\partial b_{j}} = \sum_i \dfrac{\partial L}{\partial \text{out}\_{i, j}} \dfrac{\partial \text{out}\_{i, j}}{\partial b_{j}}
$$

还是继续带入 $\dfrac{\partial L}{\partial \text{out}\_{i, j}} = \text{dout}\_{i, j}$，$\dfrac{\partial \text{out}\_{i, j}}{\partial b_j} = 1$，那么：

$$
\dfrac{\partial L}{\partial b_{j}} = \sum_i \text{dout}\_{i, j}
$$

代码直接三行结束。

```Python
def affine_backward(dout, cache):
    &#34;&#34;&#34;
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    &#34;&#34;&#34;
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_shape = x.shape
    x_reshaped = x.reshape(x_shape[0], -1)

    dx = dout.dot(w.T).reshape(x_shape)
    dw = x_reshaped.T.dot(dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
```
验证：

```Python
# Test the affine_backward function
np.random.seed(231)
x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

_, cache = affine_forward(x, w, b)
dx, dw, db = affine_backward(dout, cache)

# The error should be around e-10 or less
print(&#39;Testing affine_backward function:&#39;)
print(&#39;dx error: &#39;, rel_error(dx_num, dx))
print(&#39;dw error: &#39;, rel_error(dw_num, dw))
print(&#39;db error: &#39;, rel_error(db_num, db))
```

```{title=&#34;Output&#34;}
Testing affine_backward function:
dx error:  1.0908199508708189e-10
dw error:  2.1752635504596857e-10
db error:  7.736978834487815e-12
```

#### TODO: relu_forward

这个就是直接套公式
$$
\text{ReLU}(x) = \max(0, x)
$$

```Python
def relu_forward(x):
    &#34;&#34;&#34;
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    &#34;&#34;&#34;
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache
```

#### TODO: relu_backward

梯度也是显然的，如果 $x &gt; 0$，导数为 $1$，否则为 $0$

```Python
def relu_backward(dout, cache):
    &#34;&#34;&#34;
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    &#34;&#34;&#34;
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * np.where(x &gt; 0, 1, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
```

#### Inline Question 1

We&#39;ve only asked you to implement ReLU, but there are a number of different activation functions that one could use in neural networks, each with its pros and cons. In particular, an issue commonly seen with activation functions is getting zero (or close to zero) gradient flow during backpropagation. Which of the following activation functions have this problem? If you consider these functions in the one dimensional case, what types of input would lead to this behaviour?
1. Sigmoid
2. ReLU
3. Leaky ReLU

Answer:

**内联问题 1:**

我们只要求你实现 ReLU，但在神经网络中可以使用许多不同的激活函数，每种都有其优缺点。特别是，激活函数中常见的一个问题是在反向传播过程中获得零（或接近零）的梯度流。以下哪些激活函数存在这个问题？如果你在一维情况下考虑这些函数，什么类型的输入会导致这种行为？

1. Sigmoid
2. ReLU
3. Leaky ReLU

**答案:**

在反向传播过程中获得零梯度流的问题通常被称为“梯度消失”问题。以下是对每个激活函数的分析：

1. **Sigmoid**: Sigmoid 函数在输入值非常大或非常小时会趋向于 0 或 1，这导致其导数接近于零。因此，Sigmoid 函数在输入值非常大或非常小时会出现梯度消失问题。

2. **ReLU**: ReLU 函数在输入值小于或等于零时，其导数为零。因此，当输入值为负数时，ReLU 会出现梯度消失问题。

3. **Leaky ReLU**: Leaky ReLU 是 ReLU 的一种变体，它在输入值小于零时，导数为一个很小的常数（而不是零）。因此，Leaky ReLU 减少了梯度消失问题，因为即使在输入值为负数时，梯度也不会完全消失。

因此，Sigmoid 和 ReLU 都可能出现梯度消失问题，而 Leaky ReLU 通过在负输入时保持一个小的梯度来缓解这个问题。


1. Sigmoid: 输入值非常大或非常小时。
2. ReLU: 输入值小于或等于零时。

检验 ReLU：

```Python
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
np.random.seed(231)
x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)

out, cache = affine_relu_forward(x, w, b)
dx, dw, db = affine_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

# Relative error should be around e-10 or less
print(&#39;Testing affine_relu_forward and affine_relu_backward:&#39;)
print(&#39;dx error: &#39;, rel_error(dx_num, dx))
print(&#39;dw error: &#39;, rel_error(dw_num, dw))
print(&#39;db error: &#39;, rel_error(db_num, db))
```

```{title=&#34;Output&#34;}
Testing affine_relu_forward and affine_relu_backward:
dx error:  6.395535042049294e-11
dw error:  8.162015570444288e-11
db error:  7.826724021458994e-12
```

#### TODO: svm_loss

和之前写过的差不多。

```Python
def svm_loss(x, y):
    &#34;&#34;&#34;
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 &lt;= y[i] &lt; C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    &#34;&#34;&#34;
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    correct_class_scores = x[np.arange(num_train), y].reshape((x.shape[0],1))
    
    margins = np.maximum(0, x - correct_class_scores &#43; 1)
    margins[np.arange(num_train), y] = 0
    
    loss = np.sum(margins) / num_train
    
    # 计算梯度
    binary = margins
    binary[margins &gt; 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum
    dx = binary / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
```
#### TODO: softmax_loss

同样和之前写过的差不多。

```Python
def softmax_loss(x, y):
    &#34;&#34;&#34;
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 &lt;= y[i] &lt; C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    &#34;&#34;&#34;
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取样本数量
    num_train = x.shape[0]
    
    # 计算softmax
    shifted_logits = x - np.max(x, axis=1, keepdims=True)  # 数值稳定性
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    
    # 计算损失
    loss = -np.sum(log_probs[np.arange(num_train), y]) / num_train
    
    # 计算梯度
    dx = probs.copy()
    dx[np.arange(num_train), y] -= 1
    dx /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
```

误差检验：

```Python
np.random.seed(231)
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)

dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
loss, dx = svm_loss(x, y)

# Test svm_loss function. Loss should be around 9 and dx error should be around the order of e-9
print(&#39;Testing svm_loss:&#39;)
print(&#39;loss: &#39;, loss)
print(&#39;dx error: &#39;, rel_error(dx_num, dx))

dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)

# Test softmax_loss function. Loss should be close to 2.3 and dx error should be around e-8
print(&#39;\nTesting softmax_loss:&#39;)
print(&#39;loss: &#39;, loss)
print(&#39;dx error: &#39;, rel_error(dx_num, dx))
```
```{title=&#34;Output&#34;}
Testing svm_loss:
loss:  8.999602749096233
dx error:  1.4021566006651672e-09

Testing softmax_loss:
loss:  2.302545844500738
dx error:  9.384673161989355e-09
```

#### TODO: TwoLayerNet.__init__

要我们训练一个 `affine - relu - affine - softmax` 的两层神经网络。`np.random.normal` 随机初始化一个高斯分布的概率密度随机数。

```Python
def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        &#34;&#34;&#34;
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        &#34;&#34;&#34;
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys &#39;W1&#39; and &#39;b1&#39; and second layer                 #
        # weights and biases using the keys &#39;W2&#39; and &#39;b2&#39;.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params[&#39;W1&#39;] = np.random.normal(0, weight_scale, (input_dim,hidden_dim))
        self.params[&#39;W2&#39;] = np.random.normal(0, weight_scale, (hidden_dim,num_classes))
        self.params[&#39;b1&#39;] = np.zeros(hidden_dim)
        self.params[&#39;b2&#39;] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
```

#### TODO: TwoLayerNet.loss

```Python
def loss(self, X, y=None):
        &#34;&#34;&#34;
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        &#34;&#34;&#34;
        scores = None

        W1 = self.params[&#39;W1&#39;]
        b1 = self.params[&#39;b1&#39;]
        W2 = self.params[&#39;W2&#39;]
        b2 = self.params[&#39;b2&#39;]

        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        h1, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(h1, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don&#39;t forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算损失和梯度
        loss, dscores = softmax_loss(scores, y)
        
        # 添加L2正则化
        loss &#43;= 0.5 * self.reg * (np.sum(W1 * W1) &#43; np.sum(W2 * W2))
        
        # 反向传播
        # 第二层的反向传播
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        # 第一层的反向传播
        dx, dW1, db1 = affine_relu_backward(dh1, cache1)
        
        # 添加正则化梯度
        dW2 &#43;= self.reg * W2
        dW1 &#43;= self.reg * W1
        
        grads = {
          &#39;W1&#39;: dW1,
          &#39;b1&#39;: db1,
          &#39;W2&#39;: dW2,
          &#39;b2&#39;: db2
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
```
检验梯度：

```Python
np.random.seed(231)
N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-3
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

print(&#39;Testing initialization ... &#39;)
W1_std = abs(model.params[&#39;W1&#39;].std() - std)
b1 = model.params[&#39;b1&#39;]
W2_std = abs(model.params[&#39;W2&#39;].std() - std)
b2 = model.params[&#39;b2&#39;]
assert W1_std &lt; std / 10, &#39;First layer weights do not seem right&#39;
assert np.all(b1 == 0), &#39;First layer biases do not seem right&#39;
assert W2_std &lt; std / 10, &#39;Second layer weights do not seem right&#39;
assert np.all(b2 == 0), &#39;Second layer biases do not seem right&#39;

print(&#39;Testing test-time forward pass ... &#39;)
model.params[&#39;W1&#39;] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params[&#39;b1&#39;] = np.linspace(-0.1, 0.9, num=H)
model.params[&#39;W2&#39;] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params[&#39;b2&#39;] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff &lt; 1e-6, &#39;Problem with test-time forward pass&#39;

print(&#39;Testing training loss (no regularization)&#39;)
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) &lt; 1e-10, &#39;Problem with training-time loss&#39;

model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) &lt; 1e-10, &#39;Problem with regularization loss&#39;

# Errors should be around e-7 or less
for reg in [0.0, 0.7]:
  print(&#39;Running numeric gradient check with reg = &#39;, reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print(&#39;%s relative error: %.2e&#39; % (name, rel_error(grad_num, grads[name])))
```

```{title=&#34;Output&#34;}
Testing initialization ... 
Testing test-time forward pass ... 
Testing training loss (no regularization)
Running numeric gradient check with reg =  0.0
W1 relative error: 1.53e-08
W2 relative error: 3.37e-10
b1 relative error: 8.01e-09
b2 relative error: 4.33e-10
Running numeric gradient check with reg =  0.7
W1 relative error: 2.53e-07
W2 relative error: 2.85e-08
b1 relative error: 1.35e-08
b2 relative error: 1.97e-09
```

#### TODO: Solver

构造 solver 训练。

```Python
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
model = TwoLayerNet(input_size, hidden_size, num_classes)
solver = None

##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves about 36% #
# accuracy on the validation set.                                            #
##############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

solver = Solver(model, data, optim_config={&#39;learning_rate&#39;: 1e-3})
solver.train()

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
```

#### TODO: hyperparameters

```Python
best_model = None


#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_model.                                                          #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on thexs previous exercises.                          #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

learning_rates = [1e-4, 1e-3, 1e-2, 3e-2]
regularization_strengths = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]

results = {}
best_val = -1
best_model = None

for lr in learning_rates:
    for reg in regularization_strengths:
        model = TwoLayerNet(input_size, hidden_size, num_classes, reg=reg)
        solver = Solver(model, data, optim_config={&#39;learning_rate&#39;: lr})
        solver.train()
        
        train_accuracy = solver.train_acc_history[-1]
        val_accuracy = solver.val_acc_history[-1]
        
        results[(lr, reg)] = (train_accuracy, val_accuracy)
        
        if val_accuracy &gt; best_val:
            best_val = val_accuracy
            best_model = model
            
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print(&#39;lr %e reg %e train accuracy: %f val accuracy: %f&#39; % (
                lr, reg, train_accuracy, val_accuracy))

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
```
跑了 11 分钟。

测试模型准确率：

```Python
y_val_pred = np.argmax(best_model.loss(data[&#39;X_val&#39;]), axis=1)
print(&#39;Validation set accuracy: &#39;, (y_val_pred == data[&#39;y_val&#39;]).mean())

y_test_pred = np.argmax(best_model.loss(data[&#39;X_test&#39;]), axis=1)
print(&#39;Test set accuracy: &#39;, (y_test_pred == data[&#39;y_test&#39;]).mean())
```

超过了 48%

```{title=&#34;Output&#34;}
Validation set accuracy:  0.509
Test set accuracy:  0.488
```

#### Inline Question 2

Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.

1. Train on a larger dataset.
2. Add more hidden units.
3. Increase the regularization strength.
4. None of the above.

${\textit Your Answer:}$

${\textit Your Explanation:}$

**内联问题 2:**

现在你已经训练了一个神经网络分类器，你可能发现测试准确率远低于训练准确率。我们可以通过哪些方法来减小这个差距？选择所有适用的选项。

1. 在更大的数据集上训练
2. 增加隐藏单元数量
3. 增加正则化强度
4. 以上都不是

${\textit Your Answer:}$ 1, 3

${\textit Your Explanation:}$

1. **在更大的数据集上训练**: 
   - 更大的训练数据集可以帮助模型学习更通用的特征
   - 减少过拟合的风险，因为模型需要适应更多样的数据

2. **增加隐藏单元数量**: 
   - 增加模型复杂度实际上会加大过拟合的风险
   - 可能会使训练和测试准确率的差距更大

3. **增加正则化强度**:
   - 正则化是专门用来减少过拟合的技术
   - 通过限制权重的大小，迫使模型学习更简单的特征表示
   - 有助于提高模型的泛化能力

### Q5: Higher Level Representations: Image Features

使用 HOG 和 color histogram 特征提取一下图像的信息，简单来说就是将图片的特征表现得更明显。

```Python
from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
```

#### TODO: Train SVM on features

用新数据训练，代码和之前差不多

```Python
# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg,
                 num_iters=2000, verbose=False)
        
        y_train_pred = svm.predict(X_train_feats)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val_feats)
        val_accuracy = np.mean(y_val == y_val_pred)
        
        results[(lr, reg)] = (train_accuracy, val_accuracy)
        
        if val_accuracy &gt; best_val:
            best_val = val_accuracy
            best_svm = svm

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print(&#39;lr %e reg %e train accuracy: %f val accuracy: %f&#39; % (
                lr, reg, train_accuracy, val_accuracy))
    
print(&#39;best validation accuracy achieved: %f&#39; % best_val)
```

```{title=&#34;Output&#34;}
lr 1.000000e-09 reg 5.000000e&#43;04 train accuracy: 0.099714 val accuracy: 0.093000
lr 1.000000e-09 reg 5.000000e&#43;05 train accuracy: 0.093898 val accuracy: 0.078000
lr 1.000000e-09 reg 5.000000e&#43;06 train accuracy: 0.414571 val accuracy: 0.413000
lr 1.000000e-08 reg 5.000000e&#43;04 train accuracy: 0.092082 val accuracy: 0.077000
lr 1.000000e-08 reg 5.000000e&#43;05 train accuracy: 0.413224 val accuracy: 0.422000
lr 1.000000e-08 reg 5.000000e&#43;06 train accuracy: 0.409714 val accuracy: 0.393000
lr 1.000000e-07 reg 5.000000e&#43;04 train accuracy: 0.417327 val accuracy: 0.421000
lr 1.000000e-07 reg 5.000000e&#43;05 train accuracy: 0.407857 val accuracy: 0.396000
lr 1.000000e-07 reg 5.000000e&#43;06 train accuracy: 0.321898 val accuracy: 0.309000
best validation accuracy achieved: 0.422000
```

```Python
# Evaluate your trained SVM on the test set: you should be able to get at least 0.40
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)
```

`0.424`

#### Inline question 1:

Describe the misclassification results that you see. Do they make sense?


${\textit Your Answer:}$ 可以理解，因为有些太相似了。


#### TODO: Neural Network on image features

代码和之前差不多。

```Python
from cs231n.classifiers.fc_net import TwoLayerNet
from cs231n.solver import Solver

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

data = {
    &#39;X_train&#39;: X_train_feats, 
    &#39;y_train&#39;: y_train, 
    &#39;X_val&#39;: X_val_feats, 
    &#39;y_val&#39;: y_val, 
    &#39;X_test&#39;: X_test_feats, 
    &#39;y_test&#39;: y_test, 
}

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

learning_rates = np.linspace(1e-2, 2.75e-2, 4)
regularization_strengths = np.geomspace(1e-6, 1e-4, 3)

results = {}
best_val = -1

import itertools

for lr, reg in itertools.product(learning_rates, regularization_strengths):
    model = TwoLayerNet(input_dim, hidden_dim, num_classes,reg = reg)
    solver = Solver(model, data, optim_config={&#39;learning_rate&#39;: lr}, num_epochs=15, verbose=False)
    solver.train()

    results[(lr, reg)] = solver.best_val_acc

    if results[(lr, reg)] &gt; best_val:
        best_val = results[(lr, reg)]
        best_net = model

# Print out results.
for lr, reg in sorted(results):
    val_accuracy = results[(lr, reg)]
    print(&#39;lr %e reg %e val accuracy: %f&#39; % (lr, reg, val_accuracy))
    
print(&#39;best validation accuracy achieved during cross-validation: %f&#39; % best_val)
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

```{title=&#34;Output&#34;}
lr 1.000000e-02 reg 1.000000e-06 val accuracy: 0.517000
lr 1.000000e-02 reg 1.000000e-05 val accuracy: 0.516000
lr 1.000000e-02 reg 1.000000e-04 val accuracy: 0.516000
lr 1.583333e-02 reg 1.000000e-06 val accuracy: 0.534000
lr 1.583333e-02 reg 1.000000e-05 val accuracy: 0.528000
lr 1.583333e-02 reg 1.000000e-04 val accuracy: 0.532000
lr 2.166667e-02 reg 1.000000e-06 val accuracy: 0.555000
lr 2.166667e-02 reg 1.000000e-05 val accuracy: 0.557000
lr 2.166667e-02 reg 1.000000e-04 val accuracy: 0.543000
lr 2.750000e-02 reg 1.000000e-06 val accuracy: 0.570000
lr 2.750000e-02 reg 1.000000e-05 val accuracy: 0.566000
lr 2.750000e-02 reg 1.000000e-04 val accuracy: 0.557000
best validation accuracy achieved during cross-validation: 0.570000
```

## Assignment 2

### Q1: Multi-Layer Fully Connected Neural Networks

#### TODO: fc_net

##### \_\_init__

先解释一下 \_\_init__ 的参数。

参数翻译：
- `hidden_dims`: 一个整数列表，指定每个隐藏层的大小（神经元数量）
- `input_dim`: 一个整数，指定输入层的维度大小（默认值为3x32x32，适用于32x32的RGB图像）
- `num_classes`: 一个整数，指定需要分类的类别数量（默认为10类）
- `dropout_keep_ratio`: 丢弃强度，一个0到1之间的标量，表示dropout保留神经元的比例。如果等于1则表示不使用dropout
- `normalization`: 指定网络使用的归一化类型，可选值包括：
    - batchnorm: 批量归一化
    - layernorm: 层归一化
    - None: 不使用归一化（默认值）
- `reg`: 一个标量，表示L2正则化的强度
- `weight_scale`: 一个标量，表示权重初始化时使用的正态分布标准差
- `dtype`: numpy数据类型对象。所有计算都将使用此数据类型：
- float32: 运算更快但精度较低
- float64: 适用于数值梯度检查，精度更高
- `seed`: 随机种子。如果不为None，则传递给dropout层使其具有确定性，便于进行梯度检查

然后开始初始化参数，

```python
        # 获取所有层的维度
        dims = [input_dim] &#43; hidden_dims &#43; [num_classes]

        # 初始化每一层的参数
        for i in range(self.num_layers):
            # 初始化权重矩阵,使用正态分布
            self.params[&#39;W&#39; &#43; str(i &#43; 1)] = weight_scale * np.random.randn(dims[i], dims[i &#43; 1])
            # 初始化偏置向量为0
            self.params[&#39;b&#39; &#43; str(i &#43; 1)] = np.zeros(dims[i &#43; 1])
            
            # 如果使用批归一化且不是最后一层，最后一层不需要正则化参数
            if self.normalization and i &lt; self.num_layers - 1:
                # gamma初始化为1
                self.params[&#39;gamma&#39; &#43; str(i &#43; 1)] = np.ones(dims[i &#43; 1])
                # beta初始化为0 
                self.params[&#39;beta&#39; &#43; str(i &#43; 1)] = np.zeros(dims[i &#43; 1])
```

注意最后一层不需要正则化参数，因为模型里最后一层是 softmax，本身就会归一化到 $0 \sim 1$

##### loss

把 assignment1 的 layers.py 先抄过来，然后写一下前向传播和反向传播。

```python
def loss(self, X, y=None):
    &#34;&#34;&#34;Compute loss and gradient for the fully connected net.
    
    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
    &#34;&#34;&#34;
    X = X.astype(self.dtype)
    mode = &#34;test&#34; if y is None else &#34;train&#34;

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
        self.dropout_param[&#34;mode&#34;] = mode
    if self.normalization == &#34;batchnorm&#34;:
        for bn_param in self.bn_params:
            bn_param[&#34;mode&#34;] = mode
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you&#39;ll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you&#39;ll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 用一个变量保存上一层的输出
    layer_input = X
    caches = {}
    # 对前面 L - 1 层进行操作，因为最后一层的操作和前面的不一样
    for i in range(1, self.num_layers):
        W = self.params[&#39;W&#39; &#43; str(i)]
        b = self.params[&#39;b&#39; &#43; str(i)]

        # 计算affine层的输出
        affine_out, affine_cache = affine_forward(layer_input, W, b)
        # 计算relu层的输出
        relu_out, relu_cache = relu_forward(affine_out)

        # 保存cache
        caches[&#39;affine_cache&#39; &#43; str(i)] = affine_cache
        caches[&#39;relu_cache&#39; &#43; str(i)] = relu_cache

        # 更新layer_input
        layer_input = relu_out

    # 最后一层的操作
    W = self.params[&#39;W&#39; &#43; str(self.num_layers)]
    b = self.params[&#39;b&#39; &#43; str(self.num_layers)]

    scores, affine_cache = affine_forward(layer_input, W, b)
    caches[&#39;affine_cache&#39; &#43; str(self.num_layers)] = affine_cache

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early.
    if mode == &#34;test&#34;:
        return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don&#39;t forget to add L2 regularization!               #
    #                                                                          #
    # When using batch/layer normalization, you don&#39;t need to regularize the   #
    # scale and shift parameters.                                              #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 计算loss
    loss, dscores = softmax_loss(scores, y)

    # 先计算最后一层的梯度
    W = self.params[&#39;W&#39; &#43; str(self.num_layers)]
    affine_cache = caches[&#39;affine_cache&#39; &#43; str(self.num_layers)]
    d_relu_out, dW, db = affine_backward(dscores, affine_cache)
    grads[&#39;W&#39; &#43; str(self.num_layers)] = dW &#43; self.reg * W
    grads[&#39;b&#39; &#43; str(self.num_layers)] = db

    # 计算前面的梯度
    for i in range(self.num_layers - 1, 0, -1):
        W = self.params[&#39;W&#39; &#43; str(i)]
        affine_cache = caches[&#39;affine_cache&#39; &#43; str(i)]
        relu_cache = caches[&#39;relu_cache&#39; &#43; str(i)]

        # 先计算relu层的梯度
        d_affine_out = relu_backward(d_relu_out, relu_cache)
        # 再计算affine层的梯度
        d_relu_out, dW, db = affine_backward(d_affine_out, affine_cache)

        # 保存梯度
        grads[&#39;W&#39; &#43; str(i)] = dW &#43; self.reg * W
        grads[&#39;b&#39; &#43; str(i)] = db
        
    # 加上正则化项
    for i in range(1, self.num_layers &#43; 1):
        W = self.params[&#39;W&#39; &#43; str(i)]
        loss &#43;= 0.5 * self.reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

```

初始化误差检查：

```
Running check with reg =  0
Initial loss:  2.3004790897684924
W1 relative error: 1.4839895075340334e-07
W2 relative error: 2.212047929031316e-05
W3 relative error: 3.5272528081494203e-07
b1 relative error: 5.376386325179258e-09
b2 relative error: 2.085654276112763e-09
b3 relative error: 5.7957243458479405e-11
Running check with reg =  3.14
Initial loss:  7.052114776533016
W1 relative error: 3.904542008453064e-09
W2 relative error: 6.86942277940646e-08
W3 relative error: 2.131129859578198e-08
b1 relative error: 1.475242847895799e-08
b2 relative error: 1.7223751746766738e-09
b3 relative error: 1.5702714832602802e-10
```

#### TODO: Use a three-layer Net to overfit 50 training examples by tweaking just the learning rate and initialization scale.

使用一个三层神经网络，仅通过调整学习率和初始化规模，对 50 个训练样本进行过拟合。

```python
# TODO: Use a three-layer Net to overfit 50 training examples by 
# tweaking just the learning rate and initialization scale.

num_train = 50
small_data = {
  &#34;X_train&#34;: data[&#34;X_train&#34;][:num_train],
  &#34;y_train&#34;: data[&#34;y_train&#34;][:num_train],
  &#34;X_val&#34;: data[&#34;X_val&#34;],
  &#34;y_val&#34;: data[&#34;y_val&#34;],
}

# weight_scale = 1e-2   # Experiment with this!
# learning_rate = 1e-4  # Experiment with this!
weight_scale = 5e-2
learning_rate = 3e-3
model = FullyConnectedNet(
    [100, 100],
    weight_scale=weight_scale,
    dtype=np.float64
)
solver = Solver(
    model,
    small_data,
    print_every=10,
    num_epochs=20,
    batch_size=25,
    update_rule=&#34;sgd&#34;,
    optim_config={&#34;learning_rate&#34;: learning_rate},
)
solver.train()

plt.plot(solver.loss_history)
plt.title(&#34;Training loss history&#34;)
plt.xlabel(&#34;Iteration&#34;)
plt.ylabel(&#34;Training loss&#34;)
plt.grid(linestyle=&#39;--&#39;, linewidth=0.5)
plt.show()
```

```
(Iteration 1 / 40) loss: 28.158169
(Epoch 0 / 20) train acc: 0.280000; val_acc: 0.113000
(Epoch 1 / 20) train acc: 0.220000; val_acc: 0.124000
(Epoch 2 / 20) train acc: 0.320000; val_acc: 0.103000
(Epoch 3 / 20) train acc: 0.620000; val_acc: 0.143000
(Epoch 4 / 20) train acc: 0.740000; val_acc: 0.136000
(Epoch 5 / 20) train acc: 0.800000; val_acc: 0.133000
(Iteration 11 / 40) loss: 0.497961
(Epoch 6 / 20) train acc: 0.960000; val_acc: 0.129000
(Epoch 7 / 20) train acc: 0.940000; val_acc: 0.130000
(Epoch 8 / 20) train acc: 0.920000; val_acc: 0.106000
(Epoch 9 / 20) train acc: 0.980000; val_acc: 0.117000
(Epoch 10 / 20) train acc: 0.980000; val_acc: 0.120000
(Iteration 21 / 40) loss: 0.045432
(Epoch 11 / 20) train acc: 1.000000; val_acc: 0.120000
(Epoch 12 / 20) train acc: 1.000000; val_acc: 0.119000
(Epoch 13 / 20) train acc: 1.000000; val_acc: 0.120000
(Epoch 14 / 20) train acc: 1.000000; val_acc: 0.120000
(Epoch 15 / 20) train acc: 1.000000; val_acc: 0.119000
(Iteration 31 / 40) loss: 0.015703
(Epoch 16 / 20) train acc: 1.000000; val_acc: 0.119000
(Epoch 17 / 20) train acc: 1.000000; val_acc: 0.120000
(Epoch 18 / 20) train acc: 1.000000; val_acc: 0.120000
(Epoch 19 / 20) train acc: 1.000000; val_acc: 0.120000
(Epoch 20 / 20) train acc: 1.000000; val_acc: 0.120000
```

![](/image/ML/CS231n/6.png)

#### TODO: Use a five-layer Net to overfit 50 training examples by tweaking just the learning rate and initialization scale.

使用一个五层神经网络，仅通过调整学习率和初始化规模，对 50 个训练样本进行过拟合。

```python
# TODO: Use a five-layer Net to overfit 50 training examples by 
# tweaking just the learning rate and initialization scale.

num_train = 50
small_data = {
  &#39;X_train&#39;: data[&#39;X_train&#39;][:num_train],
  &#39;y_train&#39;: data[&#39;y_train&#39;][:num_train],
  &#39;X_val&#39;: data[&#39;X_val&#39;],
  &#39;y_val&#39;: data[&#39;y_val&#39;],
}

# learning_rate = 2e-3  # Experiment with this!
# weight_scale = 1e-5   # Experiment with this!
learning_rate = 1e-3
weight_scale = 1e-1
model = FullyConnectedNet(
    [100, 100, 100, 100],
    weight_scale=weight_scale,
    dtype=np.float64
)
solver = Solver(
    model,
    small_data,
    print_every=10,
    num_epochs=20,
    batch_size=25,
    update_rule=&#39;sgd&#39;,
    optim_config={&#39;learning_rate&#39;: learning_rate},
)
solver.train()

plt.plot(solver.loss_history)
plt.title(&#39;Training loss history&#39;)
plt.xlabel(&#39;Iteration&#39;)
plt.ylabel(&#39;Training loss&#39;)
plt.grid(linestyle=&#39;--&#39;, linewidth=0.5)
plt.show()
```
```
(Iteration 1 / 40) loss: 146.090563
(Epoch 0 / 20) train acc: 0.140000; val_acc: 0.109000
(Epoch 1 / 20) train acc: 0.140000; val_acc: 0.107000
(Epoch 2 / 20) train acc: 0.320000; val_acc: 0.121000
(Epoch 3 / 20) train acc: 0.680000; val_acc: 0.109000
(Epoch 4 / 20) train acc: 0.920000; val_acc: 0.130000
(Epoch 5 / 20) train acc: 0.940000; val_acc: 0.138000
(Iteration 11 / 40) loss: 0.118771
(Epoch 6 / 20) train acc: 0.980000; val_acc: 0.129000
(Epoch 7 / 20) train acc: 0.980000; val_acc: 0.135000
(Epoch 8 / 20) train acc: 1.000000; val_acc: 0.130000
(Epoch 9 / 20) train acc: 1.000000; val_acc: 0.130000
(Epoch 10 / 20) train acc: 1.000000; val_acc: 0.130000
(Iteration 21 / 40) loss: 0.000431
(Epoch 11 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 12 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 13 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 14 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 15 / 20) train acc: 1.000000; val_acc: 0.131000
(Iteration 31 / 40) loss: 0.000366
(Epoch 16 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 17 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 18 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 19 / 20) train acc: 1.000000; val_acc: 0.131000
(Epoch 20 / 20) train acc: 1.000000; val_acc: 0.131000
```
![](/image/ML/CS231n/7.png)

#### Inline Question 1

Did you notice anything about the comparative difficulty of training the three-layer network vs. training the five-layer network? In particular, based on your experience, which network seemed more sensitive to the initialization scale? Why do you think that is the case?

Answer:
[FILL THIS IN]

你注意到训练三层网络与训练五层网络在难度上的比较了吗？具体来说，根据你的经验，哪个网络对初始化规模更敏感？你认为为什么会这样？

答案：五层的更难。原因如下：

1. 梯度消失/爆炸问题: 由于五层网络更深,信号需要传播更多层,使得梯度在反向传播时更容易出现消失或爆炸。如果初始化尺度不合适,这个问题会更加严重。

2. 参数规模: 五层网络的参数数量更多,需要一个更合适的初始化尺度来保持各层激活值在合理范围内。初始化尺度过大或过小都会导致训练困难。

3. 优化难度: 更深的网络意味着更复杂的损失曲面,对初始点的选择(由初始化决定)更加敏感。不恰当的初始化可能使网络陷入不良的局部最优。


到目前为止,我们一直使用的是普通的随机梯度下降(SGD)作为更新规则。更复杂的更新规则可以使深度网络的训练变得更容易。我们将实现几个最常用的更新规则,并将它们与普通的 SGD 进行比较。
具体可以看[官方讲义](https://cs231n.github.io/neural-networks-3/#sgd)

#### TODO: sgd_momentum

根据讲义公式

```python
v = mu * v - learning_rate * dx # integrate velocity
x &#43;= v # integrate position
```

```python
def sgd_momentum(w, dw, config=None):
    &#34;&#34;&#34;
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    &#34;&#34;&#34;
    if config is None:
        config = {}
    config.setdefault(&#34;learning_rate&#34;, 1e-2)
    config.setdefault(&#34;momentum&#34;, 0.9)
    v = config.get(&#34;velocity&#34;, np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    v = config[&#34;momentum&#34;] * v - config[&#34;learning_rate&#34;] * dw
    next_w = w &#43; v

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config[&#34;velocity&#34;] = v

    return next_w, config
```

误差

```
next_w error:  8.882347033505819e-09
velocity error:  4.269287743278663e-09
```
对比

![](/image/ML/CS231n/8.png)

#### TODO: RMSProp

根据讲义公式

```python
cache = decay_rate * cache &#43; (1 - decay_rate) * dx**2
x &#43;= - learning_rate * dx / (np.sqrt(cache) &#43; eps)
```

```python
def rmsprop(w, dw, config=None):
    &#34;&#34;&#34;
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    &#34;&#34;&#34;
    if config is None:
        config = {}
    config.setdefault(&#34;learning_rate&#34;, 1e-2)
    config.setdefault(&#34;decay_rate&#34;, 0.99)
    config.setdefault(&#34;epsilon&#34;, 1e-8)
    config.setdefault(&#34;cache&#34;, np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don&#39;t forget to update cache value stored in    #
    # config[&#39;cache&#39;].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache = config[&#34;cache&#34;]
    cache = config[&#34;decay_rate&#34;] * cache &#43; (1 - config[&#34;decay_rate&#34;]) * dw ** 2
    next_w = w - config[&#34;learning_rate&#34;] * dw / (np.sqrt(cache) &#43; config[&#34;epsilon&#34;])
    config[&#34;cache&#34;] = cache

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
```

误差

```
next_w error:  9.524687511038133e-08
cache error:  2.6477955807156126e-09
```

#### TODO: Adam

根据讲义公式
```python
# t is your iteration counter going from 1 to infinity
m = beta1*m &#43; (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2*v &#43; (1-beta2)*(dx**2)
vt = v / (1-beta2**t)
x &#43;= - learning_rate * mt / (np.sqrt(vt) &#43; eps)
```

```python
def adam(w, dw, config=None):
    &#34;&#34;&#34;
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    &#34;&#34;&#34;
    if config is None:
        config = {}
    config.setdefault(&#34;learning_rate&#34;, 1e-3)
    config.setdefault(&#34;beta1&#34;, 0.9)
    config.setdefault(&#34;beta2&#34;, 0.999)
    config.setdefault(&#34;epsilon&#34;, 1e-8)
    config.setdefault(&#34;m&#34;, np.zeros_like(w))
    config.setdefault(&#34;v&#34;, np.zeros_like(w))
    config.setdefault(&#34;t&#34;, 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don&#39;t forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    t = config[&#34;t&#34;] &#43; 1
    m = config[&#34;beta1&#34;] * config[&#34;m&#34;] &#43; (1 - config[&#34;beta1&#34;]) * dw
    mt = m / (1 - config[&#34;beta1&#34;] ** t)
    v = config[&#34;beta2&#34;] * config[&#34;v&#34;] &#43; (1 - config[&#34;beta2&#34;]) * dw ** 2
    vt = v / (1 - config[&#34;beta2&#34;] ** t)
    next_w = w - config[&#34;learning_rate&#34;] * mt / (np.sqrt(vt) &#43; config[&#34;epsilon&#34;])

    config[&#34;t&#34;] = t
    config[&#34;m&#34;] = m
    config[&#34;v&#34;] = v

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
```
误差

```
next_w error:  1.1395691798535431e-07
v error:  4.208314038113071e-09
m error:  4.214963193114416e-09
```

整体对比：

![](/image/ML/CS231n/9.png)

#### Inline Question 2

AdaGrad, like Adam, is a per-parameter optimization method that uses the following update rule:

```
cache &#43;= dw**2
w &#43;= - learning_rate * dw / (np.sqrt(cache) &#43; eps)
```

John notices that when he was training a network with AdaGrad that the updates became very small, and that his network was learning slowly. Using your knowledge of the AdaGrad update rule, why do you think the updates would become very small? Would Adam have the same issue?


Answer: 
[FILL THIS IN]

AdaGrad和Adam一样,是一种基于每个参数的优化方法,它使用以下更新规则:

```
cache &#43;= dw**2
w &#43;= - learning_rate * dw / (np.sqrt(cache) &#43; eps)
```

John注意到当他使用AdaGrad训练网络时,更新变得非常小,他的网络学习速度变慢。根据你对AdaGrad更新规则的理解,你认为为什么更新会变得很小?Adam会有同样的问题吗?

Answer: 

AdaGrad的更新会变得很小的原因是:

1. cache是单调递增的 - 因为它不断累加平方梯度(dw**2),这些都是非负值
2. 随着训练的进行,cache会越来越大
3. 由于更新规则中cache在分母位置(w &#43;= -lr * dw / sqrt(cache)),cache的增大会导致更新步长不断减小
4. 最终会导致参数更新几乎停滞,模型难以继续学习

Adam不会有这个问题,因为:

1. Adam使用动量和RMSprop的思想,对梯度的一阶矩和二阶矩都采用指数移动平均
2. 这意味着旧的梯度信息会逐渐&#34;衰减&#34;,而不是像AdaGrad那样永久累积
3. 因此Adam能够保持相对稳定的更新步长,避免了学习完全停滞的问题

这就是为什么Adam通常比AdaGrad表现更好,特别是在训练深度神经网络时。

#### TODO: Train a Good Model!

建一个四层每层 $100$ 的神经网络。

```python
best_model = None

################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# find batch/layer normalization and dropout useful. Store your best model in  #
# the best_model variable.                                                     #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
model = FullyConnectedNet(
    [100, 100, 100],
    weight_scale=5e-2
)
solver = Solver(
    model,
    data,
    num_epochs=10,
    batch_size=100,
    update_rule=&#34;adam&#34;,
    optim_config={&#34;learning_rate&#34;: 1e-3},
    verbose=True
)
solver.train()

best_model = model

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
```
输出
```
(Iteration 1 / 4900) loss: 12.396409
(Epoch 0 / 10) train acc: 0.112000; val_acc: 0.093000
(Iteration 11 / 4900) loss: 3.950364
(Iteration 21 / 4900) loss: 2.959177
(Iteration 31 / 4900) loss: 2.403476
(Iteration 41 / 4900) loss: 2.503231
(Iteration 51 / 4900) loss: 2.397866
(Iteration 61 / 4900) loss: 2.213649
(Iteration 71 / 4900) loss: 2.026688
(Iteration 81 / 4900) loss: 1.767392
(Iteration 91 / 4900) loss: 2.077030
(Iteration 101 / 4900) loss: 2.052979
(Iteration 111 / 4900) loss: 1.921003
(Iteration 121 / 4900) loss: 1.927804
(Iteration 131 / 4900) loss: 1.933639
(Iteration 141 / 4900) loss: 1.899896
(Iteration 151 / 4900) loss: 1.943097
(Iteration 161 / 4900) loss: 1.765048
(Iteration 171 / 4900) loss: 1.771318
(Iteration 181 / 4900) loss: 1.850234
(Iteration 191 / 4900) loss: 1.610974
(Iteration 201 / 4900) loss: 1.875304
(Iteration 211 / 4900) loss: 1.746618
(Iteration 221 / 4900) loss: 1.655409
(Iteration 231 / 4900) loss: 1.810486
...
(Iteration 4871 / 4900) loss: 1.150006
(Iteration 4881 / 4900) loss: 1.142224
(Iteration 4891 / 4900) loss: 1.431774
(Epoch 10 / 10) train acc: 0.545000; val_acc: 0.495000
```
准确率，验证集 50% 达标了。

```
Validation set accuracy:  0.502
Test set accuracy:  0.483
```
### Q2: Batch Normalization

[参考论文](https://arxiv.org/pdf/1502.03167)

核心公式：

![](/image/ML/CS231n/bn1.png)

![](/image/ML/CS231n/bn2.png)

#### TODO: batchnorm_forward

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    &#34;&#34;&#34;
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean &#43; (1 - momentum) * sample_mean
    running_var = momentum * running_var &#43; (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: &#39;train&#39; or &#39;test&#39;; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    &#34;&#34;&#34;
    mode = bn_param[&#34;mode&#34;]
    eps = bn_param.get(&#34;eps&#34;, 1e-5)
    momentum = bn_param.get(&#34;momentum&#34;, 0.9)

    N, D = x.shape
    running_mean = bn_param.get(&#34;running_mean&#34;, np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get(&#34;running_var&#34;, np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == &#34;train&#34;:
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mean, var = np.mean(x, axis=0), np.var(x, axis=0)
        x_norm = (x - mean) / np.sqrt(var &#43; eps)
        out = gamma * x_norm &#43; beta

        running_mean = momentum * running_mean &#43; (1 - momentum) * mean
        running_var = momentum * running_var &#43; (1 - momentum) * var

        cache = (x, x_norm, mean, var, gamma, beta, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == &#34;test&#34;:
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_norm = (x - running_mean) / np.sqrt(running_var &#43; eps)  # 归一化
        out = gamma * x_norm &#43; beta  # 计算输出

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError(&#39;Invalid forward batchnorm mode &#34;%s&#34;&#39; % mode)

    # Store the updated running means back into bn_param
    bn_param[&#34;running_mean&#34;] = running_mean
    bn_param[&#34;running_var&#34;] = running_var

    return out, cache
```

验证：

```
Before batch normalization:
  means: [ -2.3814598  -13.18038246   1.91780462]
  stds:  [27.18502186 34.21455511 37.68611762]

After batch normalization (gamma=1, beta=0)
  means: [ 1.33226763e-17 -3.94129174e-17  3.29597460e-17]
  stds:  [0.99999999 1.         1.        ]

After batch normalization (gamma= [1. 2. 3.] , beta= [11. 12. 13.] )
  means: [11. 12. 13.]
  stds:  [0.99999999 1.99999999 2.99999999]
```

```
After batch normalization (test-time):
  means: [-0.03927354 -0.04349152 -0.10452688]
  stds:  [1.01531428 1.01238373 0.97819988]
```

#### TODO: batchnorm_backward

计算图太麻烦了，直接用链式求导。

```python
def batchnorm_backward(dout, cache):
    &#34;&#34;&#34;
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    &#34;&#34;&#34;
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, mean, var, gamma, beta, eps = cache

    N, D = x.shape
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var &#43; eps) ** -1.5, axis=0)
    dmean = np.sum(dx_norm * -1 / np.sqrt(var &#43; eps), axis=0) &#43; dvar * np.mean(-2 * (x - mean), axis=0)
    dx = dx_norm / np.sqrt(var &#43; eps) &#43; dvar * 2 * (x - mean) / N &#43; dmean / N

    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
```

```
dx error:  1.7029261167605239e-09
dgamma error:  7.420414216247087e-13
dbeta error:  2.8795057655839487e-12
```

#### TODO: layer_utils

在每个 ReLU 激活函数前添加批量归一化层。

```python
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    affine_out,affine_cache = affine_forward(x, w, b)
    bn_out,bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    relu_out,relu_cache = relu_forward(bn_out)
    cache = (affine_cache, bn_cache, relu_cache)
    return relu_out, cache

def affine_bn_relu_backward(dout, cache):
    affine_cache, bn_cache, relu_cache = cache
    drelu_out = relu_backward(dout, relu_cache)
    dbn_out, dgamma, dbeta = batchnorm_backward(drelu_out, bn_cache)
    dx, dw, db = affine_backward(dbn_out, affine_cache)
    return dx, dw, db, dgamma, dbeta

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```
验证：

```
Running check with reg =  0
Initial loss:  2.2611955101340957
W1 relative error: 1.10e-04
W2 relative error: 2.85e-06
W3 relative error: 3.92e-10
b1 relative error: 2.22e-03
b2 relative error: 2.22e-08
b3 relative error: 4.78e-11
beta1 relative error: 7.33e-09
beta2 relative error: 1.07e-09
gamma1 relative error: 7.47e-09
gamma2 relative error: 2.41e-09

Running check with reg =  3.14
Initial loss:  6.996533220108303
W1 relative error: 1.98e-06
W2 relative error: 2.28e-06
W3 relative error: 1.11e-08
b1 relative error: 2.78e-09
b2 relative error: 2.22e-08
b3 relative error: 2.23e-10
beta1 relative error: 6.32e-09
beta2 relative error: 5.69e-09
gamma1 relative error: 5.94e-09
gamma2 relative error: 4.14e-09
```

![](/image/ML/CS231n/10.png)

![](/image/ML/CS231n/11.png)

![](/image/ML/CS231n/12.png)

#### Inline Question 1:
Describe the results of this experiment. How does the weight initialization scale affect models with/without batch normalization differently, and why?

根据图表结果，我可以帮你分析批量归一化(Batch Normalization)对权重初始化尺度的影响:

从实验结果可以观察到以下几点:

1. **无批量归一化的网络**:
- 对权重初始化尺度非常敏感
- 当初始化尺度过大或过小时,性能都会显著下降
- 只在一个很窄的权重初始化范围内表现良好

2. **有批量归一化的网络**:
- 对权重初始化尺度的依赖性明显降低
- 在很宽的权重初始化范围内都能保持稳定的性能
- 即使在较大的初始化尺度下也能达到较好的训练和验证准确率

3. **原因分析**:
- 批量归一化通过标准化每一层的输出,使得网络层之间的数据分布保持稳定
- 这种标准化效果减弱了初始权重带来的影响,因为无论初始权重如何,经过批量归一化后的输出都会被调整到类似的分布
- 这使得网络训练更加稳定,不容易受到权重初始化的影响

这个实验很好地展示了批量归一化的一个重要优势:它能够降低网络对权重初始化的敏感度,使得训练更加稳定和鲁棒。


#### Inline Question 2:
Describe the results of this experiment. What does this imply about the relationship between batch normalization and batch size? Why is this relationship observed?

描述本次实验的结果。这对批量归一化和批量大小之间的关系有何启示？为何会观察到这种关系？

从实验结果图中我们可以观察到以下几点:

1. 当使用较小的批量大小(batch_size=5,10)时,批量归一化的性能明显下降,训练和验证准确率都较低且波动较大。

2. 当使用较大的批量大小(batch_size=50)时,批量归一化表现最好,训练更稳定,准确率更高。

这说明批量归一化的效果与批量大小有很强的相关性,原因是:

1. 批量归一化依赖于每个mini-batch内的统计量(均值和方差)来进行归一化。当批量太小时:
   - 计算的统计量波动较大,不能很好地代表整体数据分布
   - 这种不稳定的归一化会影响网络训练

2. 较大的批量大小可以:
   - 提供更稳定可靠的统计估计
   - 使归一化效果更接近整体数据分布
   - 减少训练过程中的噪声

3. 这也解释了为什么在实际应用中,批量归一化通常需要相对较大的batch size(如32或64)才能发挥最佳效果。

这个实验结果强调了在使用批量归一化时需要合理选择批量大小,以在计算效率和归一化效果之间取得平衡。

#### layer normalization

[参考论文](https://arxiv.org/pdf/1607.06450)

层归一化，简单来说，就是不受 batch_size 的影响。

#### TODO: layernorm_forward

```python
def layernorm_forward(x, gamma, beta, ln_param):
    &#34;&#34;&#34;
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    &#34;&#34;&#34;
    out, cache = None, None
    eps = ln_param.get(&#34;eps&#34;, 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 沿特征维度计算均值和方差（axis=1）
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    # 归一化处理
    x_norm = (x - mean) / np.sqrt(var &#43; eps)
    
    # 缩放和平移
    out = gamma * x_norm &#43; beta
    
    # 缓存反向传播需要的中间变量
    cache = (x, x_norm, mean, var, gamma, beta, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache
```
验证：

```
Before layer normalization:
  means: [-59.06673243 -47.60782686 -43.31137368 -26.40991744]
  stds:  [10.07429373 28.39478981 35.28360729  4.01831507]

After layer normalization (gamma=1, beta=0)
  means: [ 4.81096644e-16  0.00000000e&#43;00  0.00000000e&#43;00 -2.96059473e-16]
  stds:  [0.99999995 0.99999999 1.         0.99999969]

After layer normalization (gamma= [3. 3. 3.] , beta= [5. 5. 5.] )
  means: [5. 5. 5. 5.]
  stds:  [2.99999985 2.99999998 2.99999999 2.99999907]
```

#### TODO: layernorm_backward

```python
def layernorm_backward(dout, cache):
    &#34;&#34;&#34;
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you&#39;ve done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    &#34;&#34;&#34;
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, mean, var, gamma, beta, eps = cache
    N, D = x.shape

    # 计算dgamma和dbeta
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    # 计算dx_norm
    dx_norm = dout * gamma
    
    # 计算方差梯度
    dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var &#43; eps)**-1.5, axis=1, keepdims=True)
    
    # 计算均值梯度
    dmean = np.sum(dx_norm * (-1) / np.sqrt(var &#43; eps), axis=1, keepdims=True) &#43; \
            dvar * np.mean(-2 * (x - mean), axis=1, keepdims=True)
    
    # 计算最终输入梯度
    dx = (dx_norm / np.sqrt(var &#43; eps)) &#43; (dvar * 2 * (x - mean) / D) &#43; (dmean / D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
```
验证：

```
dx error:  1.433615657860454e-09
dgamma error:  4.519489546032799e-12
dbeta error:  2.276445013433725e-12
```

对比：

![](/image/ML/CS231n/13.png)

#### Inline Question 3:
Which of these data preprocessing steps is analogous to batch normalization, and which is analogous to layer normalization?

1. Scaling each image in the dataset, so that the RGB channels for each row of pixels within an image sums up to 1.
2. Scaling each image in the dataset, so that the RGB channels for all pixels within an image sums up to 1.  
3. Subtracting the mean image of the dataset from each image in the dataset.
4. Setting all RGB values to either 0 or 1 depending on a given threshold.

这些数据预处理步骤中，哪一步与批量归一化类似，哪一步与层归一化类似？

1. 对数据集中的每张图像进行缩放，确保图像中每行像素的RGB通道之和为1。
2. 对数据集中的每幅图像进行缩放，使图像中所有像素的RGB通道之和为1。
3. 从数据集中的每幅图像中减去数据集的平均图像。
4. 根据给定的阈值，将所有RGB值设置为0或1。

3对应批量归一化，2对应层归一化。批量归一化通过减去均值进行中心化（如选项3），而层归一化在样本内所有特征上归一化（如选项2对整个图像做缩放）。

#### Inline Question 4:
When is layer normalization likely to not work well, and why?

1. Using it in a very deep network
2. Having a very small dimension of features
3. Having a high regularization term

层归一化在什么时候可能效果不佳，原因是什么？
1. 在非常深的网络中使用它
2. 特征维度非常小
3. 正则化项取值很高


是 2.当特征维度非常小时，层归一化可能效果不佳。因为层归一化需要在单个样本的所有特征维度上计算统计量（均值和方差），当特征维度很小时：
- 统计量的估计会变得不稳定
- 归一化操作可能过度缩放特征，导致信息丢失
- 特别是当特征维度为1时，归一化后所有特征值会变为0，完全破坏原始数据
相比之下，在特征维度较大的情况下，统计量的估计更可靠，归一化效果更好。其他选项与层归一化的有效性没有直接关联。

### Q3: Dropout

[参考论文](https://arxiv.org/pdf/1207.0580)

简单来说就是前向传播的时候会随机把一些神经元的值变为 0，可以缓解过拟合。

![](/image/ML/CS231n/dropout.jpeg)

#### TODO: dropout_forward

参考官网讲义：[官网讲义](https://cs231n.github.io/neural-networks-2/#reg)

生成一个 0/1 概率向量，其中概率为 $p$，如果概率小于 $p$ 则**不会**被置为 $0$，为了最后输出期望统一要乘上 $\dfrac{1}{p}$ 放缩。

```python
def dropout_forward(x, dropout_param):
    &#34;&#34;&#34;
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: &#39;test&#39; or &#39;train&#39;. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    &#34;&#34;&#34;
    p, mode = dropout_param[&#34;p&#34;], dropout_param[&#34;mode&#34;]
    if &#34;seed&#34; in dropout_param:
        np.random.seed(dropout_param[&#34;seed&#34;])

    mask = None
    out = None

    if mode == &#34;train&#34;:
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) &lt; p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == &#34;test&#34;:
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache
```

```
Running tests with p =  0.25
Mean of input:  10.000207878477502
Mean of train-time output:  10.014059116977283
Mean of test-time output:  10.000207878477502
Fraction of train-time output set to zero:  0.749784
Fraction of test-time output set to zero:  0.0

Running tests with p =  0.4
Mean of input:  10.000207878477502
Mean of train-time output:  9.977917658761159
Mean of test-time output:  10.000207878477502
Fraction of train-time output set to zero:  0.600796
Fraction of test-time output set to zero:  0.0

Running tests with p =  0.7
Mean of input:  10.000207878477502
Mean of train-time output:  9.987811912159426
Mean of test-time output:  10.000207878477502
Fraction of train-time output set to zero:  0.30074
Fraction of test-time output set to zero:  0.0
```

#### TODO: dropout_backward

```python
def dropout_backward(dout, cache):
    &#34;&#34;&#34;
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    &#34;&#34;&#34;
    dropout_param, mask = cache
    mode = dropout_param[&#34;mode&#34;]

    dx = None
    if mode == &#34;train&#34;:
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == &#34;test&#34;:
        dx = dout
    return dx
```

```
dx relative error:  5.44560814873387e-11
```

#### Inline Question 1:
What happens if we do not divide the values being passed through inverse dropout by `p` in the dropout layer? Why does that happen?

如果我们在 dropout 层中没有将通过反向 dropout 的值除以 $p$，会发生什么？为什么会发生这种情况？

Answer:
如果在 dropout 层中没有通过 $p$ 来除以传递的值，那么在训练和测试阶段的输出分布将不一致。在训练阶段，dropout 会随机将一些神经元的输出设置为零，并通过 $p$ 来缩放剩余的输出，以保持激活的期望值不变。然而，如果不进行这种缩放，训练阶段的输出将会比测试阶段的输出小 $p$ 倍。这会导致模型在训练和测试阶段表现不一致，从而影响模型的泛化能力。

#### TODO: 给 fc_net 加上 dropout

```python
np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for dropout_keep_ratio in [1, 0.75, 0.5]:
    print(&#39;Running check with dropout = &#39;, dropout_keep_ratio)
    model = FullyConnectedNet(
        [H1, H2],
        input_dim=D,
        num_classes=C,
        weight_scale=5e-2,
        dtype=np.float64,
        dropout_keep_ratio=dropout_keep_ratio,
        seed=123
    )

    loss, grads = model.loss(X, y)
    print(&#39;Initial loss: &#39;, loss)

    # Relative errors should be around e-6 or less.
    # Note that it&#39;s fine if for dropout_keep_ratio=1 you have W2 error be on the order of e-5.
    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
        print(&#39;%s relative error: %.2e&#39; % (name, rel_error(grad_num, grads[name])))
    print()
```

```
Running check with dropout =  1
Initial loss:  2.3004790897684924
W1 relative error: 1.48e-07
W2 relative error: 2.21e-05
W3 relative error: 3.53e-07
b1 relative error: 5.38e-09
b2 relative error: 2.09e-09
b3 relative error: 5.80e-11

Running check with dropout =  0.75
Initial loss:  2.302371489704412
W1 relative error: 1.90e-07
W2 relative error: 4.76e-06
W3 relative error: 2.60e-08
b1 relative error: 4.73e-09
b2 relative error: 1.82e-09
b3 relative error: 1.70e-10

Running check with dropout =  0.5
Initial loss:  2.3042759220785896
W1 relative error: 3.11e-07
W2 relative error: 1.84e-08
W3 relative error: 5.35e-08
b1 relative error: 5.37e-09
b2 relative error: 2.99e-09
b3 relative error: 1.13e-10
```

训练对比：

![](/image/ML/CS231n/14.png)

#### Inline Question 2:
Compare the validation and training accuracies with and without dropout -- what do your results suggest about dropout as a regularizer?

比较使用和不使用 dropout 的验证和训练准确率——你的结果对 dropout 作为正则化器有什么建议？

Answer:

实验结果显示，使用dropout（keep_ratio=0.25）时：
1. 训练准确率略低于不使用dropout的情况，说明dropout通过随机失活神经元降低了模型对训练数据的过拟合能力
2. 验证准确率显著高于不使用dropout的情况，且与训练准确率的差距更小，表明模型泛化能力更好
3. 验证曲线更平滑稳定，说明dropout起到了正则化作用，有效抑制了过拟合现象

这说明dropout通过阻止神经元间的协同适应（co-adaptation），迫使网络学习更鲁棒的特征，从而提升模型在未见数据上的表现。

### Q4: Convolutional Neural Networks

#### TODO: conv_forward_naive

```python
def conv_forward_naive(x, w, b, conv_param):
    &#34;&#34;&#34;
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - &#39;stride&#39;: The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - &#39;pad&#39;: The number of pixels that will be used to zero-pad the input.


    During padding, &#39;pad&#39; zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H&#39;, W&#39;) where H&#39; and W&#39; are given by
      H&#39; = 1 &#43; (H &#43; 2 * pad - HH) / stride
      W&#39; = 1 &#43; (W &#43; 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    &#34;&#34;&#34;
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 先获取一些需要用到的数据
    N, C, H_input, W_input = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param[&#34;stride&#34;]  # 步长
    pad = conv_param[&#34;pad&#34;]  # 填充数量

    # 计算卷积后的高和宽
    out_H = int(1 &#43; (H_input &#43; 2 * pad - HH) / stride)
    out_W = int(1 &#43; (W_input &#43; 2 * pad - WW) / stride)

    # 给x的上下左右填充上pad个0
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), &#34;constant&#34;, constant_values=0)
    # 将卷积核w转换成F * (C * HH * WW)的矩阵 (便于使用矩阵乘法)
    w_row = w.reshape(F, -1)
    # 生成空白输出便于后续循环填充
    out = np.zeros((N, F, out_H, out_W))

    # 开始卷积
    for n in range(N):  # 遍历样本
        for f in range(F):  # 遍历卷积核
            for i in range(out_H):  # 遍历高
                for j in range(out_W):  # 遍历宽
                    # 获取当前卷积窗口
                    window = x_pad[n, :, i * stride:i * stride &#43; HH, j * stride:j * stride &#43; WW]
                    # 将卷积窗口拉成一行
                    window_row = window.reshape(1, -1)
                    # 计算当前卷积窗口和卷积核的卷积结果
                    out[n, f, i, j] = np.sum(window_row * w_row[f, :]) &#43; b[f]
      
	  # 将pad后的x存入cache (省的反向传播的时候在计算一次)
    x = x_pad

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache
```

```
Testing conv_forward_naive
difference:  2.2121476417505994e-08
```

![](/image/ML/CS231n/15.png)

#### TODO: conv_backward_naive

```python
def conv_backward_naive(dout, cache):
    &#34;&#34;&#34;
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    &#34;&#34;&#34;
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取一些需要用到的数据
    x, w, b, conv_param = cache
    N, C, H_input, W_input = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param[&#34;stride&#34;]  # 步长
    pad = conv_param[&#34;pad&#34;]  # 填充数量

    # 计算卷积后的高和宽
    out_H = int(1 &#43; (H_input - HH) / stride)
    out_W = int(1 &#43; (W_input - WW) / stride)

    # 给dx,dw,db分配空间
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前卷积窗口
                    window = x[n, :, i * stride:i * stride &#43; HH, j * stride:j * stride &#43; WW]
                    # 计算db
                    db[f] &#43;= dout[n, f, i, j]
                    # 计算dw
                    dw[f] &#43;= window * dout[n, f, i, j]
                    # 计算dx
                    dx[n, :, i * stride:i * stride &#43; HH, j * stride:j * stride &#43; WW] &#43;= w[f] * dout[n, f, i, j]

    # 去掉dx的pad
    dx = dx[:, :, pad:H_input - pad, pad:W_input - pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
```

```
Testing conv_backward_naive function
dx error:  1.159803161159293e-08
dw error:  2.2471264748452487e-10
db error:  3.37264006649648e-11
```

#### TODO: max_pool_forward_naive

```python
def max_pool_forward_naive(x, pool_param):
    &#34;&#34;&#34;
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - &#39;pool_height&#39;: The height of each pooling region
      - &#39;pool_width&#39;: The width of each pooling region
      - &#39;stride&#39;: The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H&#39;, W&#39;) where H&#39; and W&#39; are given by
      H&#39; = 1 &#43; (H - pool_height) / stride
      W&#39; = 1 &#43; (W - pool_width) / stride
    - cache: (x, pool_param)
    &#34;&#34;&#34;
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取一些需要用到的数据
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    pool_height = pool_param[&#34;pool_height&#34;]  # 池化核高
    pool_width = pool_param[&#34;pool_width&#34;]  # 池化核宽
    stride = pool_param[&#34;stride&#34;]  # 步长

    # 计算池化后的高和宽
    out_H = int(1 &#43; (H - pool_height) / stride)
    out_W = int(1 &#43; (W - pool_width) / stride)

    # 给out分配空间
    out = np.zeros((N, C, out_H, out_W))

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前池化窗口
                    window = x[n, c, i * stride:i * stride &#43; pool_height, j * stride:j * stride &#43; pool_width]
                    # 计算当前池化窗口的最大值
                    out[n, c, i, j] = np.max(window)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache
```

```
Testing max_pool_forward_naive function:
difference:  4.1666665157267834e-08
```

#### TOOD: max_pool_backward_naive

```python
def max_pool_backward_naive(dout, cache):
    &#34;&#34;&#34;
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    &#34;&#34;&#34;
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取一些需要用到的数据
    x, pool_param = cache
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    pool_height = pool_param[&#34;pool_height&#34;]  # 池化核高
    pool_width = pool_param[&#34;pool_width&#34;]  # 池化核宽
    stride = pool_param[&#34;stride&#34;]  # 步长

    # 计算池化后的高和宽
    out_H = int(1 &#43; (H - pool_height) / stride)
    out_W = int(1 &#43; (W - pool_width) / stride)

    # 给dx分配空间
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前池化窗口
                    window = x[n, c, i * stride:i * stride &#43; pool_height, j * stride:j * stride &#43; pool_width]
                    # 计算当前池化窗口的最大值
                    max_index = np.argmax(window)
                    # 计算dx
                    dx[n, c, i * stride &#43; max_index // pool_width, j * stride &#43; max_index % pool_width] &#43;= dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
```

```
Testing max_pool_backward_naive function:
dx error:  3.27562514223145e-12
```

#### Fast Layers

`im2col_cython.pyx` 最上方加一行代码： `#cython: language_level=2`

再改一下启动脚本，运行。

```python
# Remember to restart the runtime after executing this cell!
%cd ./cs231n
!python setup.py build_ext --inplace
%cd ../
```

中途报了 `nameerror: name &#39;col2im_6d_cython&#39; is not defined` 的错误，重启一下笔记本就好了。

```
Testing conv_forward_fast:
Naive: 4.879596s
Fast: 0.010973s
Speedup: 444.682390x
Difference:  1.970563140655889e-11

Testing conv_backward_fast:
Naive: 6.945447s
Fast: 0.002007s
Speedup: 3459.776366x
dx difference:  9.43434568725122e-12
dw difference:  4.420587653909754e-13
db difference:  3.481354613192702e-14

Testing pool_forward_fast:
Naive: 0.302344s
fast: 0.004001s
speedup: 75.559852x
difference:  0.0

Testing pool_backward_fast:
Naive: 0.315859s
fast: 0.010774s
speedup: 29.317090x
dx difference:  0.0
```

```
Testing conv_relu_pool
dx error:  4.397502834267091e-09
dw error:  3.651699397290073e-09
db error:  7.054812624223923e-10

Testing conv_relu:
dx error:  8.03522627181292e-09
dw error:  2.0902405745264502e-10
db error:  3.287958402642519e-10
```

#### TODO: Three-Layer Convolutional Network

```python
class ThreeLayerConvNet(object):
    &#34;&#34;&#34;
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    &#34;&#34;&#34;

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        &#34;&#34;&#34;
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        &#34;&#34;&#34;
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys &#39;W1&#39; and &#39;b1&#39;; use keys &#39;W2&#39; and &#39;b2&#39; for the       #
        # weights and biases of the hidden affine layer, and keys &#39;W3&#39; and &#39;b3&#39;    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        C, H, W = input_dim  # 获取输入数据的通道数，高度，宽度

        # 卷积层
        self.params[&#34;W1&#34;] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params[&#34;b1&#34;] = np.zeros(num_filters)

        # 全连接层
        self.params[&#34;W2&#34;] = np.random.normal(0, weight_scale, (num_filters * H * W // 4, hidden_dim))
        self.params[&#34;b2&#34;] = np.zeros(hidden_dim)

        # 全连接层
        self.params[&#34;W3&#34;] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params[&#34;b3&#34;] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        &#34;&#34;&#34;
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        &#34;&#34;&#34;
        W1, b1 = self.params[&#34;W1&#34;], self.params[&#34;b1&#34;]
        W2, b2 = self.params[&#34;W2&#34;], self.params[&#34;b2&#34;]
        W3, b3 = self.params[&#34;W3&#34;], self.params[&#34;b3&#34;]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {&#34;stride&#34;: 1, &#34;pad&#34;: (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {&#34;pool_height&#34;: 2, &#34;pool_width&#34;: 2, &#34;stride&#34;: 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)  # 卷积层
        out2, cache2 = affine_relu_forward(out1, W2, b2)  # 全连接层
        scores, cache3 = affine_forward(out2, W3, b3)  # 全连接层

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don&#39;t forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算损失
        loss, dout = softmax_loss(scores, y)
        loss &#43;= 0.5 * self.reg * (np.sum(W1 ** 2) &#43; np.sum(W2 ** 2) &#43; np.sum(W3 ** 2))  # L2正则化

        # 计算梯度
        dout, grads[&#34;W3&#34;], grads[&#34;b3&#34;] = affine_backward(dout, cache3)  # 全连接层
        dout, grads[&#34;W2&#34;], grads[&#34;b2&#34;] = affine_relu_backward(dout, cache2)  # 全连接层
        dout, grads[&#34;W1&#34;], grads[&#34;b1&#34;] = conv_relu_pool_backward(dout, cache1)  # 卷积层

        # 加上正则化项的梯度
        grads[&#34;W3&#34;] &#43;= self.reg * W3
        grads[&#34;W2&#34;] &#43;= self.reg * W2
        grads[&#34;W1&#34;] &#43;= self.reg * W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
```

Sanity Check Loss

```
Initial loss (no regularization):  2.302586071243987
Initial loss (with regularization):  2.508255638232932
```

Gradient Check

```
W1 max relative error: 1.380104e-04
W2 max relative error: 1.822723e-02
W3 max relative error: 3.064049e-04
b1 max relative error: 3.477652e-05
b2 max relative error: 2.516375e-03
b3 max relative error: 7.945660e-10
```

Overfit Small Data

```
Small data training accuracy: 0.82

Small data validation accuracy: 0.252
```

![](/image/ML/CS231n/16.png)

Train the Network

```
Full data training accuracy: 0.4761836734693878

Full data validation accuracy: 0.499
```

Visualize Filters

![](/image/ML/CS231n/17.png)

spatial batchnorm 和 spatial groupnorm 不太会，抄个代码鸽一下。

#### TODO: spatial_batchnorm_forward

```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    &#34;&#34;&#34;
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: &#39;train&#39; or &#39;test&#39;; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    &#34;&#34;&#34;
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    x = np.moveaxis(x, 1, -1).reshape(-1, C)  # 将C通道放到最后，然后reshape成二维数组
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)  # 调用batchnorm_forward
    out = np.moveaxis(out.reshape(N, H, W, C), -1, 1)  # 将C通道放到第二维，然后reshape成四维数组

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache
```

```
Before spatial batch normalization:
  shape:  (2, 3, 4, 5)
  means:  [9.33463814 8.90909116 9.11056338]
  stds:  [3.61447857 3.19347686 3.5168142 ]
After spatial batch normalization:
  shape:  (2, 3, 4, 5)
  means:  [ 6.18949336e-16  5.99520433e-16 -1.22124533e-16]
  stds:  [0.99999962 0.99999951 0.9999996 ]
After spatial batch normalization (nontrivial gamma, beta):
  shape:  (2, 3, 4, 5)
  means:  [6. 7. 8.]
  stds:  [2.99999885 3.99999804 4.99999798]
```

```
After spatial batch normalization (test-time):
  means:  [-0.08034406  0.07562881  0.05716371  0.04378383]
  stds:  [0.96718744 1.0299714  1.02887624 1.00585577]
```

#### TODO: spatial_batchnorm_backward

```python
def spatial_batchnorm_backward(dout, cache):
    &#34;&#34;&#34;
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    &#34;&#34;&#34;
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape  # N个样本，C个通道，H高，W宽
    dout = np.moveaxis(dout, 1, -1).reshape(-1, C)  # 将C通道放到最后，然后reshape成二维数组
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)  # 调用batchnorm_backward
    dx = np.moveaxis(dx.reshape(N, H, W, C), -1, 1)  # 将C通道放到第二维，然后reshape成四维数组

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
```

```
dx error:  2.786648197756335e-07
dgamma error:  7.0974817113608705e-12
dbeta error:  3.275608725278405e-12
```

#### TODO: spatial_groupnorm_forward

```python
def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    &#34;&#34;&#34;
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    &#34;&#34;&#34;
    out, cache = None, None
    eps = gn_param.get(&#34;eps&#34;, 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽

    # 将C通道分成G组，每组有C//G个通道
    x = x.reshape(N, G, C // G, H, W)  # reshape成五维数组
    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)  # 求均值
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True)  # 求方差
    x_norm = (x - x_mean) / np.sqrt(x_var &#43; eps)  # 归一化

    x_norm = x_norm.reshape(N, C, H, W)  # reshape成四维数组
    out = gamma * x_norm &#43; beta  # 伸缩平移

    cache = (x, x_norm, x_mean, x_var, gamma, beta, G, eps)  # 缓存变量

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache
```

```
Before spatial group normalization:
  shape:  (2, 6, 4, 5)
  means:  [9.72505327 8.51114185 8.9147544  9.43448077]
  stds:  [3.67070958 3.09892597 4.27043622 3.97521327]
After spatial group normalization:
  shape:  (2, 6, 4, 5)
  means:  [-2.14643118e-16  5.25505565e-16  2.65528340e-16 -3.38618023e-16]
  stds:  [0.99999963 0.99999948 0.99999973 0.99999968]
```

#### TODO: spatial_groupnorm_backward

```python
def spatial_groupnorm_backward(dout, cache):
    &#34;&#34;&#34;
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    &#34;&#34;&#34;
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, x_mean, x_var, gamma, beta, G, eps = cache  # 从缓存中取出变量
    N, C, H, W = dout.shape  # N个样本，C个通道，H高，W宽

    # 计算dgamma和dbeta
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)  # 求dgamma
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # 求dbeta

    # 准备数据
    x = x.reshape(N, G, C // G, H, W)  # reshape成五维数组

    m = C // G * H * W
    dx_norm = (dout * gamma).reshape(N, G, C // G, H, W)
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * np.power((x_var &#43; eps), -1.5), axis=(2, 3, 4), keepdims=True)
    dx_mean = np.sum(dx_norm * (-1) / np.sqrt(x_var &#43; eps), axis=(2, 3, 4), keepdims=True) &#43; dx_var * np.sum(-2 * (x - x_mean), axis=(2, 3, 4),
                                                                                                             keepdims=True) / m
    dx = dx_norm / np.sqrt(x_var &#43; eps) &#43; dx_var * 2 * (x - x_mean) / m &#43; dx_mean / m
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
```

```
dx error:  7.413109648400194e-08
dgamma error:  9.468195772749234e-12
dbeta error:  3.354494437653335e-12
```

### Q5: PyTorch on CIFAR-10

使用 PyTorch 来实现一些神经网络。

#### Barebones PyTorch: Two-Layer Network

实现一个两层 ReLU 的全连接神经网络，主要是对 PyTorch 的基本语法熟悉一下。

```python
import torch.nn.functional as F  # useful stateless functions

def two_layer_fc(x, params):
    &#34;&#34;&#34;
    A fully-connected neural networks; the architecture is:
    NN is fully connected -&gt; ReLU -&gt; fully connected layer.
    Note that this function only defines the forward pass; 
    PyTorch will take care of the backward pass for us.
    
    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.
    
    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).
    
    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    &#34;&#34;&#34;
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]
    
    w1, w2 = params
    
    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don&#39;t need to keep references to intermediate values.
    # you can also use `.clamp(min=0)`, equivalent to F.relu()
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x
    

def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 50), dtype=dtype)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    scores = two_layer_fc(x, [w1, w2])
    print(scores.size())  # you should see [64, 10]

two_layer_fc_test()
```
`torch.nn.functional` 是定义了一个无状态函数，他提供了一系列的常用函数。

`torch.zeros` 是 PyTorch 中的一个函数，用于创建一个全为零的张量。

`flatten` 是将输入的图像数据扁平化为一维向量，以便于后续的全连接层处理。

`mm()` 矩阵乘法。

#### TODO: Barebones PyTorch: Three-Layer ConvNet

完成函数 three_layer_convnet 的实现，该函数将执行三层卷积网络的前向传播。网络应具有以下架构：
1. 一个卷积层（带偏置），具有 `channel_1` 个滤波器，每个滤波器的形状为 `KW1 x KH1`，并且有零填充为2。
2. ReLU 非线性激活。
3. 一个卷积层（带偏置），具有 `channel_2` 个滤波器，每个滤波器的形状为 `KW2 x KH2`，并且有零填充为1。
4. ReLU 非线性激活。
5. 一个全连接层（带偏置），生成 C 类的分数。

```python
def three_layer_convnet(x, params):
    &#34;&#34;&#34;
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    &#34;&#34;&#34;
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 第一层卷积
    x = F.conv2d(x, conv_w1, conv_b1, padding=2)  # 使用零填充
    x = F.relu(x)  # ReLU 激活

    # 第二层卷积
    x = F.conv2d(x, conv_w2, conv_b2, padding=1)  # 使用零填充
    x = F.relu(x)  # ReLU 激活

    # 展平
    x = flatten(x)

    # 全连接层
    scores = x.mm(fc_w) &#43; fc_b  # 计算分数

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores
```

#### Barebones PyTorch: Initialization

`random_weight(shape)` 使用 Kaiming 正规化方法初始化权重张量。

`zero_weight(shape)` 初始化一个全为零的权重张量。对于实例化偏置参数非常有用。

```python
def random_weight(shape):
    &#34;&#34;&#34;
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    &#34;&#34;&#34;
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

# create a weight of shape [3 x 5]
# you should see the type `torch.cuda.FloatTensor` if you use GPU. 
# Otherwise it should be `torch.FloatTensor`
random_weight((3, 5))
```

#### TODO: BareBones PyTorch: Training a ConvNet

##### BareBones PyTorch: 训练卷积神经网络

训练一个三层卷积网络。网络应具有以下架构：

1. 卷积层（带偏置），使用32个5x5的滤波器，零填充为2
2. ReLU激活
3. 卷积层（带偏置），使用16个3x3的滤波器，零填充为1
4. ReLU激活
5. 全连接层（带偏置），用于计算10个类别的分数

您应该使用上面定义的`random_weight`函数来初始化权重矩阵，并使用`zero_weight`函数来初始化偏置向量。

您不需要调整任何超参数，但如果一切正常，您应该在一个epoch后达到超过42%的准确率。

```python
learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

################################################################################
# TODO: Initialize the parameters of a three-layer ConvNet.                    #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight(channel_1)
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight(channel_2)
fc_w = random_weight((channel_2 * 32 * 32, 10))
fc_b = zero_weight(10)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)
```

```{title=&#34;Output&#34;}
Iteration 0, loss = 3.8646
Checking accuracy on the val set
Got 107 / 1000 correct (10.70%)

Iteration 100, loss = 1.8770
Checking accuracy on the val set
Got 317 / 1000 correct (31.70%)

Iteration 200, loss = 1.8188
Checking accuracy on the val set
Got 375 / 1000 correct (37.50%)

Iteration 300, loss = 1.6608
Checking accuracy on the val set
Got 397 / 1000 correct (39.70%)

Iteration 400, loss = 1.7184
Checking accuracy on the val set
Got 437 / 1000 correct (43.70%)

Iteration 500, loss = 1.6697
Checking accuracy on the val set
Got 450 / 1000 correct (45.00%)

Iteration 600, loss = 1.6106
...
Iteration 700, loss = 1.4275
Checking accuracy on the val set
Got 438 / 1000 correct (43.80%)
```
#### TODO: Module API: Three-Layer ConvNet

##### 模块 API：三层卷积网络

实现一个三层卷积网络，后面跟一个全连接层。网络架构应该与第二部分相同：

1. 使用 `channel_1` 个 5x5 的卷积层，零填充为 2
2. ReLU 激活
3. 使用 `channel_2` 个 3x3 的卷积层，零填充为 1
4. ReLU 激活
5. 全连接层，输出 `num_classes` 个类别

你应该使用 Kaiming 正态初始化方法来初始化模型的权重矩阵。

**提示**: [PyTorch 文档](http://pytorch.org/docs/stable/nn.html#conv2d)

在你实现三层卷积网络后，`test_ThreeLayerConvNet` 函数将运行你的实现；它应该打印输出分数的形状为 `(64, 10)`。

```python
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)  # 第一层卷积
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)  # 第二层卷积
        self.fc3 = nn.Linear(channel_2 * 32 * 32, num_classes)  # 全连接层
        
        # Kaiming 正态初始化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                          END OF YOUR CODE                            #       
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 定义前向传播
        x = F.relu(self.conv1(x))  # 第一层卷积 &#43; ReLU
        x = F.relu(self.conv2(x))  # 第二层卷积 &#43; ReLU
        x = flatten(x)  # 展平
        scores = self.fc3(x)  # 全连接层

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores


def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]
test_ThreeLayerConvNet()
```

#### TODO: Module API: Train a Three-Layer ConvNet

```python
learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None
################################################################################
# TODO: Instantiate your ThreeLayerConvNet model and a corresponding optimizer #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# 实例化三层卷积神经网络模型
model = ThreeLayerConvNet(in_channel=3, channel_1=channel_1, channel_2=channel_2, num_classes=10)

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

train_part34(model, optimizer)
```

```{title=&#34;Output&#34;}
Iteration 0, loss = 2.8420
Checking accuracy on validation set
Got 120 / 1000 correct (12.00)

Iteration 100, loss = 1.8090
Checking accuracy on validation set
Got 336 / 1000 correct (33.60)

Iteration 200, loss = 1.8638
Checking accuracy on validation set
Got 385 / 1000 correct (38.50)

Iteration 300, loss = 1.5297
Checking accuracy on validation set
Got 402 / 1000 correct (40.20)

Iteration 400, loss = 1.5403
Checking accuracy on validation set
Got 430 / 1000 correct (43.00)

Iteration 500, loss = 1.5430
Checking accuracy on validation set
Got 450 / 1000 correct (45.00)

Iteration 600, loss = 1.5708
...
Iteration 700, loss = 1.6809
Checking accuracy on validation set
Got 466 / 1000 correct (46.60)
```

#### TODO: Sequential API: Three-Layer ConvNet

##### 顺序API：三层卷积神经网络

使用 `nn.Sequential` 来定义和训练一个三层卷积神经网络，其架构与我们在第三部分中使用的相同：

1. 卷积层（带偏置）使用32个5x5的滤波器，零填充为2
2. ReLU
3. 卷积层（带偏置）使用16个3x3的滤波器，零填充为1
4. ReLU
5. 全连接层（带偏置）计算10个类别的分数

您可以使用默认的PyTorch权重初始化。

您应该使用带有Nesterov动量0.9的随机梯度下降来优化您的模型。

同样，您不需要调整任何超参数，但您应该在训练一个周期后看到准确率超过55%。

```python
channel_1 = 32
channel_2 = 16
learning_rate = 1e-2

model = None
optimizer = None

################################################################################
# TODO: Rewrite the 3-layer ConvNet with bias from Part III with the           #
# Sequential API.                                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model = nn.Sequential(
    nn.Conv2d(3, channel_1, kernel_size=5, padding=2),  # 第一层卷积
    nn.ReLU(),                                           # ReLU 激活
    nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1),  # 第二层卷积
    nn.ReLU(),                                           # ReLU 激活
    Flatten(),                                          # 展平
    nn.Linear(channel_2 * 32 * 32, 10)                 # 全连接层
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

train_part34(model, optimizer)
```

```{title=&#34;Output&#34;}
Iteration 0, loss = 2.2989
Checking accuracy on validation set
Got 116 / 1000 correct (11.60)

Iteration 100, loss = 1.6947
Checking accuracy on validation set
Got 457 / 1000 correct (45.70)

Iteration 200, loss = 1.7063
Checking accuracy on validation set
Got 476 / 1000 correct (47.60)

Iteration 300, loss = 1.1563
Checking accuracy on validation set
Got 495 / 1000 correct (49.50)

Iteration 400, loss = 1.3090
Checking accuracy on validation set
Got 553 / 1000 correct (55.30)

Iteration 500, loss = 1.0429
Checking accuracy on validation set
Got 539 / 1000 correct (53.90)

Iteration 600, loss = 1.4302
...
Iteration 700, loss = 1.3101
Checking accuracy on validation set
Got 579 / 1000 correct (57.90)
```

#### TODO: Part V. CIFAR-10 open-ended challenge

在这一部分中，您可以尝试在 CIFAR-10 上实验任何卷积神经网络架构。

现在轮到您实验架构、超参数、损失函数和优化器，以训练一个在 CIFAR-10 **验证集**上达到 **至少 70%** 准确率的模型，训练时间不超过 10 个周期。您可以使用上面的 check_accuracy 和 train 函数。您可以使用 `nn.Module` 或 `nn.Sequential` API。

使用 ResNet 模型。

```python
import torchvision
################################################################################
# TODO:                                                                        #         
# Experiment with any architectures, optimizers, and hyperparameters.          #
# Achieve AT LEAST 70% accuracy on the *validation set* within 10 epochs.      #
#                                                                              #
# Note that you can use the check_accuracy function to evaluate on either      #
# the test set or the validation set, by passing either loader_test or         #
# loader_val as the second argument to check_accuracy. You should not touch    #
# the test set until you have finished your architecture and  hyperparameter   #
# tuning, and only run the test set once at the end to report a final value.   #
################################################################################
model = None
optimizer = None

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

################################################################################
# TODO:                                                                        #         
# Experiment with any architectures, optimizers, and hyperparameters.          #
# Achieve AT LEAST 70% accuracy on the *validation set* within 10 epochs.      #
#                                                                              #
# Note that you can use the check_accuracy function to evaluate on either      #
# the test set or the validation set, by passing either loader_test or         #
# loader_val as the second argument to check_accuracy. You should not touch    #
# the test set until you have finished your architecture and  hyperparameter   #
# tuning, and only run the test set once at the end to report a final value.   #
################################################################################
model = None
optimizer = None

# 使用ResNet-18架构
model = torchvision.models.resnet18(pretrained=False)
# 修改第一层卷积（原ImageNet输入为3通道224x224，CIFAR-10为3通道32x32）
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# 移除原最大池化层（因图像尺寸较小）
model.maxpool = nn.Identity()
# 修改全连接层输出为10类
model.fc = nn.Linear(512, 10)

# 使用Adam优化器，加入学习率调度
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# You should get at least 70% accuracy.
# You may modify the number of epochs to any number below 15.
train_part34(model, optimizer, epochs=10)
```

```{title=&#34;Output&#34;}
Iteration 300, loss = 0.0958
Checking accuracy on validation set
Got 803 / 1000 correct (80.30)
```
## Assignment 3

### Q1: Image Captioning with Vanilla RNNs

RNN 结构模型图如下：

![](/image/ML/CS231n/rnn1.webp)

特点就是可以保留历史信息，其中 $x$ 可以代表一个单词向量，$x_t$ 是第 $t$ 个单词向量 (也叫做 $t$ 时刻)，图中的 $W, U, V$ 是每个时刻共用的参数。
模型公式：

$$
o_t = g(V \cdot s_t) \\\\
s_t = f(U \cdot x_t &#43; W \cdot s_{t - 1})
$$

在 CS231n 中，基本模型如下：

![](/image/ML/CS231n/rnn2.png)

先从单步模型看起：

#### TODO: rnn_step_forward

单步的前向传播直接套用公式，这里激活函数一般是 $\tanh$ 

$$
\text{next}_h = \tanh(\text{prev}_h \cdot W_h &#43; x \cdot W_x &#43; b)
$$

```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    &#34;&#34;&#34;Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    &#34;&#34;&#34;
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    next_h = np.tanh(np.dot(prev_h, Wh) &#43; np.dot(x, Wx) &#43; b)
    cache = (x, prev_h, Wx, Wh, b, next_h)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache
```

#### TODO: rnn_step_backward

反向传播我们先来看一下 $\mathrm{d}x$

根据公式 $\text{next}_h = \tanh(\text{prev}_h \cdot W_h &#43; x \cdot W_x &#43; b)$

我们先设 $z = \text{prev}_h \cdot W_h &#43; x \cdot W_x &#43; b$

也就是 $\text{next}_h = \tanh(z)$

那么 $x$ 的梯度就是
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{ \partial z} \cdot \frac{\partial z}{ \partial x} = \frac{\partial L}{ \partial z} \cdot W_x ^ {\top}
$$

那来看一下 $\dfrac{\partial L}{\partial z}$ 这是什么：

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \text{next}_h} \cdot \frac{\partial \text{next}_h}{\partial z} = \mathrm{d} \text{next}_h \cdot \frac{\partial \text{next}_h}{\partial z}
$$

那么 $\dfrac{\partial \text{next}_h}{\partial z}$ 就是对 $\tanh(z)$ 求导，导数为 $1 - \tanh^2(z)$

所以 $\dfrac{\partial L}{\partial z} = \mathrm{d} \text{next}_h (1 - \tanh^2(z))$，把这个记为 $\mathrm{d}\tanh$

所以 $\mathrm{d}x = \mathrm{d}\tanh \cdot W_x ^ {\top}$

其他参数都是同理的。

```python
def rnn_step_backward(dnext_h, cache):
    &#34;&#34;&#34;Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    &#34;&#34;&#34;
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, prev_h, Wx, Wh, b, next_h = cache
    dtanh = dnext_h * (1 - next_h**2)
    dx = np.dot(dtanh, Wx.T)
    dprev_h = np.dot(dtanh, Wh.T)
    dWx = np.dot(x.T, dtanh)
    dWh = np.dot(prev_h.T, dtanh)
    db = np.sum(dtanh, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db
```

#### TODO: rnn_forward

整体的前向传播只需要把 $\text{prev}_h$ 代入就可以了，注意保留过程。

```python
def rnn_forward(x, h0, Wx, Wh, b):
    &#34;&#34;&#34;Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    - cache: Values needed in the backward pass
    &#34;&#34;&#34;
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, T, D = x.shape
    H = h0.shape[1]
    h = np.zeros((N, T, H))
    cache = []
    prev_h = h0
    
    for t in range(T):
        prev_h, cache_t = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = prev_h
        cache.append(cache_t)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache
```

#### TODO: rnn_backward

整体反向传播需要注意的第一个是逆序的，还要注意梯度传播有两个方向：一个是当前传播下来的梯度，另一个是由下一个时间节点传播下来的梯度，加起来就好。

```python
def rnn_backward(dh, cache):
    &#34;&#34;&#34;Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
    
    NOTE: &#39;dh&#39; contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you&#39;ll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    &#34;&#34;&#34;
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    dprev_h = np.zeros((N, H))

    for t in reversed(range(T)):
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh[:, t, :] &#43; dprev_h, cache[t])
        dx[:, t, :] = dx_t
        dWx &#43;= dWx_t
        dWh &#43;= dWh_t
        db &#43;= db_t

    dh0 = dprev_h

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db
```

#### TODO: word_embedding_forward

word embedding 就是词嵌入，简单来说就是把单词表示成向量的形式，假设用最简单的独热编码 (one-hot)，这个在之前 softmax 里提到过，就是每个不同种类的单词构成的单位矩阵。

在这里，单词矩阵 $W$ 是一个 $V \times D$ 的矩阵，其中 $V$ 代表单词个数，$D$ 代表维度。$X$ 是一个 $N \times T$ 的矩阵，$N$ 代表 batch，$T$ 代表这句话有 $T$ 个词，$T$ 序列中的每个值是 $W$ 中的索引。

所以前向传播直接 W[x] 自动索引就行。

```python
def word_embedding_forward(x, W):
    &#34;&#34;&#34;Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 &lt;= idx &lt; V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    &#34;&#34;&#34;
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy&#39;s array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = W[x]
    cache = (x, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache
```

#### TODO: word_embedding_backward

首先注意提示，先学习 np.add.at()

```python
import numpy as np

# 创建一个数组
a = np.array([1, 2, 3, 4])

# 在指定索引处增加值
np.add.at(a, [0, 1, 2, 2], 1)

# 输出结果
print(a) # 结果为 [2, 3, 5, 4]
```

这个就是给数组一个列表，在指定索引处增加值。

dout 是大小 $N \times T \times D$ 的上游梯度，在 dW 矩阵上根据 $x$ 矩阵作为下标加上 dout 的值，因为 out 只依赖于 $W$ 在特定位置（即 $x$ 的元素所表示的 $W$ 的下标）的值， out 对 $W$ 求导之后系数是 $1$，所以只要在特定位置加上 dout 的值就行。


```python
def word_embedding_backward(dout, cache):
    &#34;&#34;&#34;Backward pass for word embeddings.
    
    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
    &#34;&#34;&#34;
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW
```

#### TODO: CaptioningRNN.loss CaptioningRNN.sample 



```python
import numpy as np

from ..rnn_layers import *


class CaptioningRNN:
    &#34;&#34;&#34;
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don&#39;t use any regularization for the CaptioningRNN.
    &#34;&#34;&#34;

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type=&#34;rnn&#34;,
        dtype=np.float32,
    ):
        &#34;&#34;&#34;
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either &#39;rnn&#39; or &#39;lstm&#39;.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        &#34;&#34;&#34;
        if cell_type not in {&#34;rnn&#34;, &#34;lstm&#34;}:
            raise ValueError(&#39;Invalid cell_type &#34;%s&#34;&#39; % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx[&#34;&lt;NULL&gt;&#34;]
        self._start = word_to_idx.get(&#34;&lt;START&gt;&#34;, None)
        self._end = word_to_idx.get(&#34;&lt;END&gt;&#34;, None)

        # Initialize word vectors
        self.params[&#34;W_embed&#34;] = np.random.randn(vocab_size, wordvec_dim)
        self.params[&#34;W_embed&#34;] /= 100

        # Initialize CNN -&gt; hidden state projection parameters
        self.params[&#34;W_proj&#34;] = np.random.randn(input_dim, hidden_dim)
        self.params[&#34;W_proj&#34;] /= np.sqrt(input_dim)
        self.params[&#34;b_proj&#34;] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {&#34;lstm&#34;: 4, &#34;rnn&#34;: 1}[cell_type]
        self.params[&#34;Wx&#34;] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params[&#34;Wx&#34;] /= np.sqrt(wordvec_dim)
        self.params[&#34;Wh&#34;] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params[&#34;Wh&#34;] /= np.sqrt(hidden_dim)
        self.params[&#34;b&#34;] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params[&#34;W_vocab&#34;] = np.random.randn(hidden_dim, vocab_size)
        self.params[&#34;W_vocab&#34;] /= np.sqrt(hidden_dim)
        self.params[&#34;b_vocab&#34;] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        &#34;&#34;&#34;
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T &#43; 1) where
          each element is in the range 0 &lt;= y[i, t] &lt; V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        &#34;&#34;&#34;
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t&#43;1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You&#39;ll need this
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params[&#34;W_proj&#34;], self.params[&#34;b_proj&#34;]

        # Word embedding matrix
        W_embed = self.params[&#34;W_embed&#34;]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params[&#34;Wx&#34;], self.params[&#34;Wh&#34;], self.params[&#34;b&#34;]

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params[&#34;W_vocab&#34;], self.params[&#34;b_vocab&#34;]

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is &lt;NULL&gt; using the mask above.     #
        #                                                                          #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        h0, cache_affine = affine_forward(features, W_proj, b_proj)
        word_vectors, cache_embed = word_embedding_forward(captions_in, W_embed)
        if self.cell_type == &#34;rnn&#34;:
            h, cache_rnn = rnn_forward(word_vectors, h0, Wx, Wh, b)
        elif self.cell_type == &#34;lstm&#34;:
            h, cache_lstm = lstm_forward(word_vectors, h0, Wx, Wh, b)
        else:
            raise ValueError(&#34;Invalid cell_type&#34;)
        scores, cache_temporal_affine = temporal_affine_forward(h, W_vocab, b_vocab)

        loss, dscores = temporal_softmax_loss(scores, captions_out, mask)

        dh, grads[&#34;W_vocab&#34;], grads[&#34;b_vocab&#34;] = temporal_affine_backward(dscores, cache_temporal_affine)
        if self.cell_type == &#34;rnn&#34;:
            dword_vectors, dh0, grads[&#34;Wx&#34;], grads[&#34;Wh&#34;], grads[&#34;b&#34;] = rnn_backward(dh, cache_rnn)
        elif self.cell_type == &#34;lstm&#34;:
            dword_vectors, dh0, grads[&#34;Wx&#34;], grads[&#34;Wh&#34;], grads[&#34;b&#34;] = lstm_backward(dh, cache_lstm)
        else:
            raise ValueError(&#34;Invalid cell_type&#34;)
        grads[&#34;W_embed&#34;] = word_embedding_backward(dword_vectors, cache_embed)
        _, grads[&#34;W_proj&#34;], grads[&#34;b_proj&#34;] = affine_backward(dh0, cache_affine)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sample(self, features, max_length=30):
        &#34;&#34;&#34;
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the &lt;START&gt;
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the &lt;START&gt; token.
        &#34;&#34;&#34;
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params[&#34;W_proj&#34;], self.params[&#34;b_proj&#34;]
        W_embed = self.params[&#34;W_embed&#34;]
        Wx, Wh, b = self.params[&#34;Wx&#34;], self.params[&#34;Wh&#34;], self.params[&#34;b&#34;]
        W_vocab, b_vocab = self.params[&#34;W_vocab&#34;], self.params[&#34;b_vocab&#34;]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the &lt;START&gt; token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an &lt;END&gt; token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you&#39;ll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        h, _ = affine_forward(features, W_proj, b_proj)
        c = np.zeros_like(h) if self.cell_type == &#34;lstm&#34; else None
        captions[:, 0] = self._start
        for t in range(1, max_length):
            word_vectors, _ = word_embedding_forward(captions[:, t-1], W_embed)
            if self.cell_type == &#34;rnn&#34;:
                h, _ = rnn_step_forward(word_vectors, h, Wx, Wh, b)
            elif self.cell_type == &#34;lstm&#34;:
                h, c, _ = lstm_step_forward(word_vectors, h, c, Wx, Wh, b)
            else:
                raise ValueError(&#34;Invalid cell_type&#34;)
            scores, _ = affine_forward(h, W_vocab, b_vocab)
            captions[:, t] = np.argmax(scores, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
```

### Q2: Image Captioning with Transformers

#### TODO: MultiHeadAttention

输入格式：

$N$ 表示 batch size，$S$ 表示源序列长度，$T$ 表示目标序列长度，$E$ 表示 embedding 维度。

- query：作为查询（query）使用的输入数据，形状为 (N, S, E)
- key：作为键（key）使用的输入数据，形状为 (N, T, E)
- value：作为值（value）使用的输入数据，形状为 (N, T, E)
- attn_mask：形状为 $(S, T)$ 的数组，其中 $mask_{i, j} = 0$ 表示源序列中的第 $i$ 个 token 不应影响目标序列中的第 $j$ 个 token

返回：

- output：形状为 $(N, S, E) 的张量，根据用 key 和 query 计算得到的注意力权重，对 value 中的数据加权组合后的结果。

```python
class MultiHeadAttention(nn.Module):
    &#34;&#34;&#34;
    A model layer which implements a simplified version of masked attention, as
    introduced by &#34;Attention Is All You Need&#34; (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    &#34;&#34;&#34;

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        &#34;&#34;&#34;
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        &#34;&#34;&#34;
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn&#39;t strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        &#34;&#34;&#34;
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        &#34;&#34;&#34;
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You&#39;ll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 1. 线性变换，得到 Q, K, V
        # 形状：(N, S, E) / (N, T, E) -&gt; (N, S, E) / (N, T, E)
        Q = self.query(query)   # (N, S, E)
        K = self.key(key)       # (N, T, E)
        V = self.value(value)   # (N, T, E)

        # 2. 拆分多头 (N, S, E) -&gt; (N, n_head, S, head_dim)
        def split_heads(x):
            N, L, E = x.shape
            return x.view(N, L, self.n_head, self.head_dim).transpose(1, 2)
            # (N, L, n_head, head_dim) -&gt; (N, n_head, L, head_dim)

        Q = split_heads(Q)  # (N, n_head, S, head_dim)
        K = split_heads(K)  # (N, n_head, T, head_dim)
        V = split_heads(V)  # (N, n_head, T, head_dim)

        # 3. 计算注意力分数 (Q @ K^T / sqrt(d_k))
        # Q: (N, n_head, S, head_dim)
        # K: (N, n_head, T, head_dim)
        # K.transpose(-2, -1): (N, n_head, head_dim, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (N, n_head, S, T)

        # 4. 应用掩码（mask），防止某些位置被关注 
        if attn_mask is not None:
            # attn_mask: (S, T) -&gt; (1, 1, S, T) 方便广播
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float(&#39;-inf&#39;))

        # 5. softmax 得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, n_head, S, T)

        # 6. dropout
        attn_weights = self.attn_drop(attn_weights)

        # 7. 加权求和得到每个头的输出
        # attn_weights: (N, n_head, S, T)
        # V: (N, n_head, T, head_dim)
        attn_output = torch.matmul(attn_weights, V)  # (N, n_head, S, head_dim)

        # 8. 合并多头 (N, n_head, S, head_dim) -&gt; (N, S, n_head, head_dim) -&gt; (N, S, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, S, E)

        # 9. 输出投影
        output = self.proj(attn_output)  # (N, S, E)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output
```

```
self_attn_output error:  0.0003772742211599121
masked_self_attn_output error:  0.0001526367643724865
attn_output error:  0.00035224630317522767
```
#### TODO: PositionalEncoding




## 参考

https://github.com/Divsigma/2020-cs213n/tree/master/cs231n

https://github.com/Na-moe/CS231n-2024/tree/main

https://github.com/Chia202/CS231n/tree/main

https://blog.csdn.net/leezed525/category_12388436.html

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlcs231n/  


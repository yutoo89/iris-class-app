from sklearn.datasets import load_iris

iris_dataset = load_iris()

# Keys of iris_dataset:
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
# print('Keys of iris_dataset: \n{}'.format(iris_dataset.keys()))

# Iris plants dataset
# --------------------

# **Data Set Characteristics:**

#     :Number of Instances: 150 (50 in each of three classes)
#     :Number of Attributes: 4 numeric, pre
# ...
# print(iris_dataset['DESCR'][:193] + '\n...')

# Target names: ['setosa' 'versicolor' 'virginica']
# print('Target names: {}'.format(iris_dataset['target_names']))

# 特徴量の説明
# Feature names:
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print('Feature names: \n{}'.format(iris_dataset['feature_names']))

# Type of data: <class 'numpy.ndarray'>
# print('Type of data: {}'.format(type(iris_dataset['data'])))

# Shape of data: (150, 4)
# print('Shape of data: {}'.format(iris_dataset['data'].shape))

# First five columns of data:
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]
# print('First five columns of data: \n{}'.format(iris_dataset['data'][:5]))

from sklearn.model_selection import train_test_split

# 全データを訓練セット75%とテストセット25%に分ける
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)
# X_train shape: (112, 4)
# y_train shape: (112,)
# print('X_train shape: {}'.format(X_train.shape))
# print('y_train shape: {}'.format(y_train.shape))

# X_test shape: (38, 4)
# y_test shape: (38,)
# print('X_test shape: {}'.format(X_test.shape))
# print('y_test shape: {}'.format(y_test.shape))

# X_trainのデータからDataFrameを作成
# iris_dataset.feature_namesの文字列を使ってカラムに名前をつける
import pandas as pd
import mglearn

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# DataFrameからscatter matrixを作成し、y_trainに従って色をつける
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker="o",hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

# k-最近傍法クラス分類アルゴリズムはKNeighborsClassifierに実装されている
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

import numpy as np

# X_new = np.array([[5, 2.9, 1, 0.2]])
# X_name.shape: (1, 4)
# print('X_name.shape: {}'.format(X_new.shape))

# prediction = knn.predict(X_new)
# Prediction: [0]
# Predicted target name: ['setosa']
# print('Prediction: {}'.format(prediction))
# print('Predicted target name: {}'.format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
# 予測結果の表示
# Test set prediction:
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
print("Test set prediction: {}".format(y_pred))

# モデルの精度の評価
# Test set score: 0.97
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

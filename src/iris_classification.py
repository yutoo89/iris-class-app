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

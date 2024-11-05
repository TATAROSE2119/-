# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import pickle
#import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


## 读取训练数据
X_train_file = pd.read_csv('1_Data_正常状态.csv', low_memory=False).iloc[0:, 1:]

## 数据预处理

# 删除无关的数据列
columns_to_drop = []
X_train_file.drop(columns=columns_to_drop, axis=1, inplace=True)
X_train = (X_train_file.values).astype('float32')


# 定义归一化函数
def normalize_data(X, X_mean, X_std):
    # 如果标准差为0，将标准差设置为1，避免除以0的情况
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std

# 计算训练数据的均值和标准差
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

# 归一化训练数据
X_train_normal = normalize_data(X_train, X_train_mean, X_train_std)


## 训练 PCA模型
"""
在 sklearn.decomposition.PCA 中，n_components 参数用于指定主成分分析（PCA）中要保留的成分数。它可以有不同的设置方式，每种方式都有其特定的含义：

整数值：如果 n_components 是一个整数（如 n_components=3），那么 PCA 将保留指定数量的主成分。在这种情况下，PCA 将根据解释方差的顺序选择前 n_components 个主成分。

浮点数（0到1之间）：如果 n_components 是一个浮点数（如 n_components=0.9），那么 PCA 将选择使得累计解释方差比（explained variance ratio）达到或超过这个值的主成分数。例如，n_components=0.9 表示选择足够的主成分以确保累计解释方差比达到90%。

None：如果 n_components=None，则所有的主成分都将被保留。这在数据降维中通常不使用，因为目的是通过减少特征数量来简化数据。
"""
pca = PCA(n_components=0.99 )  # 设定PCA累计贡献率
pca.fit(X_train_normal)  # 使用训练数据集对PCA进行训练

# 保存 PCA 模型到文件
with open('pca_model.pkl', 'wb') as file:
    pickle.dump(pca, file)
    
## 采用SPE和T2设置检测阈值
"""
SPE（Squared Prediction Error）
Squared Prediction Error (SPE)，也称为Q统计量，是用于监控过程偏差的统计量。SPE用于衡量观测值和其预测值之间的差异，以识别潜在的异常或过程变化。

定义：SPE 是原始数据点和其在主成分分析（PCA）重构空间中的投影点之间的欧氏距离的平方和。
用途：主要用于监控过程的稳定性和检测异常。SPE值越大，表示数据点偏离正常状态越远，可能是异常的。

T²（Hotelling’s T-squared）
Hotelling’s T-squared (T²) 是一种多变量统计量，用于衡量一个观测点与总体均值之间的离散程度。T²统计量广泛应用于多变量过程监控中，用于检测潜在的异常。
定义：T² 是基于协方差矩阵加权的观测值与其均值之间的距离。
用途：T²用于监控变量间的线性关系和整体过程的变化。高于某个阈值的T²值表示观测点可能是异常的。
"""
## 使用训练数据计算SPE阈值和T2阈值

# 对训练数据进行PCA降维
X_train_pca = pca.transform(X_train_normal)

# 计算训练数据的逆PCA
X_train_reconstructed = pca.inverse_transform(X_train_pca)

# 计算训练数据和测试数据的SPE值
train_SPE = np.sum((X_train_normal - X_train_reconstructed)**2, axis=1)

# 计算训练数据和测试数据的T2值
train_T2 = np.sum(X_train_pca**2 / pca.explained_variance_, axis=1)

# SPE和T2置信度
alpha = 0.99             

# 计算SPE的核密度估计
spe_kde = stats.gaussian_kde(train_SPE)

# 创建一个从train_SPE最小值到最大值的等间距数组
spe_range = np.linspace(np.min(train_SPE), np.max(train_SPE), 1000)

# 计算SPE范围内每个点的密度值
spe_density = spe_kde.evaluate(spe_range)

# 计算SPE的置信区间上限
# np.cumsum(spe_density) 计算累积密度
# np.sum(spe_density) 是总密度
# np.cumsum(spe_density)/np.sum(spe_density) - alpha 计算累积密度和置信度的差值
# np.argmin(np.abs(...)) 找到最接近alpha的累积密度
SPE_threshold = spe_range[np.argmin(np.abs(np.cumsum(spe_density)/np.sum(spe_density) - alpha))]

# 计算T2的核密度估计
t2_kde = stats.gaussian_kde(train_T2)

# 创建一个从train_T2最小值到最大值的等间距数组
t2_range = np.linspace(np.min(train_T2), np.max(train_T2), 1000)

# 计算T2范围内每个点的密度值
t2_density = t2_kde.evaluate(t2_range)

# 计算T2的置信区间上限
# np.cumsum(t2_density) 计算累积密度
# np.sum(t2_density) 是总密度
# np.cumsum(t2_density) / np.sum(t2_density) - alpha 计算累积密度和置信度的差值
# np.argmin(np.abs(...)) 找到最接近alpha的累积密度
T2_threshold = t2_range[np.argmin(np.abs(np.cumsum(t2_density) / np.sum(t2_density) - alpha))]

# 保存均值、标准差、两个阈值到一个文件
np.savez('X_train_params.npz', mean=X_train_mean, std=X_train_std, SPE=SPE_threshold ,T2=T2_threshold)

# 加载 PCA 模型
with open('pca_model.pkl', 'rb') as file:
    pca_loaded = pickle.load(file)
    
# 加载均值、标准差、两个阈值
params = np.load('X_train_params.npz')
X_train_mean, X_train_std, SPE_threshold, T2_threshold = params['mean'], params['std'], params['SPE'], params['T2']

## 读取测试数据，选取相应的测试数据
X_test_file = pd.read_csv('5_Data_异常状态4.csv', low_memory=False).iloc[0:, 1:]

## 数据预处理

# 删除无关的数据列
columns_to_drop = []
X_test_file.drop(columns=columns_to_drop, axis=1, inplace=True)
X_test = (X_test_file.values).astype('float32')

# 定义归一化函数
def normalize_data(X, X_mean, X_std):
    # 如果标准差为0，将标准差设置为1，避免除以0的情况
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std


# 归一化测试数据
X_test_normal = normalize_data(X_test, X_train_mean, X_train_std)

    
# 对训练数据和测试数据进行PCA降维
X_test_pca = pca_loaded.transform(X_test_normal)

# 计算训练数据和测试数据的逆PCA
X_test_reconstructed = pca_loaded.inverse_transform(X_test_pca)

# 计算训练数据和测试数据的SPE值
test_SPE = np.sum((X_test_normal - X_test_reconstructed)**2, axis=1)

# 计算训练数据和测试数据的T2值
test_T2 = np.sum(X_test_pca**2 / pca_loaded.explained_variance_, axis=1)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 真实标签（前5000个样本是正常数据，后面的是异常数据）
num_samples = X_test_normal.shape[0]
true_labels = np.concatenate([np.ones(5000), -np.ones(num_samples - 5000)])

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar=False, linewidths=0.5, linecolor='black', annot_kws={"size": 16}, ax=ax)
    for text in ax.texts:
        text.set_text(text.get_text() + "%")

    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', rotation=0, labelsize=16)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# 使用SPE值T2值和阈值来预测异常
predictions_SPE = np.where(test_SPE > SPE_threshold, -1, 1)
predictions_T2 = np.where(test_T2 > T2_threshold, -1, 1)

# 绘制混淆矩阵
class_names = ['Abnormal', 'Normal']
plot_confusion_matrix(true_labels, predictions_SPE, class_names, title='SPE_Confusion Matrix', save_path='AE_SPE_confusion_matrix.png')
plot_confusion_matrix(true_labels, predictions_T2, class_names, title='T2_Confusion Matrix', save_path='AE_T2_confusion_matrix.png')

# 绘制测试集SPE曲线
plt.figure(figsize=(8, 6))
plt.plot(test_SPE, color='#1f77b4', label='SPE', linewidth=0.5)
plt.axhline(y=SPE_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('SPE', fontsize=16)
plt.legend(fontsize=14)
plt.title('SPE Curve for Test Data', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('SPE.png', dpi=600)
plt.show()

# 绘制测试集T2曲线
plt.figure(figsize=(8, 6))
plt.plot(test_T2, color='#1f77b4', label='T2', linewidth=0.5)
plt.axhline(y=T2_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('T2', fontsize=16)
plt.legend(fontsize=14)
plt.title('T2 Curve for Test Data', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('T2.png', dpi=600)
plt.show()

# 选择要查看的样本索引
sample_index = 8200

# 获取该样本的特征值
sample_features = X_test_normal[sample_index, :]

# 计算该样本的逆PCA
sample_reconstructed = X_test_reconstructed[sample_index, :]

# 计算该样本的SPE值
sample_SPE_contributions = (sample_features - sample_reconstructed) ** 2

# 计算该样本的T2值
sample_pca = X_test_pca[sample_index, :]
sample_T2_contributions = sample_pca ** 2 / pca_loaded.explained_variance_

# 将 T2 的贡献度映射回原始特征空间
components = pca_loaded.components_
original_feature_contributions_T2 = np.dot(sample_T2_contributions, components ** 2)

# 绘制特征贡献度条形图（SPE）
plt.figure(figsize=(10, 6))
plt.bar(range(len(sample_SPE_contributions)), sample_SPE_contributions, color='#1f77b4')
plt.xlabel('Feature Index', fontsize=14)
plt.ylabel('SPE Contribution', fontsize=14)
plt.title(f'SPE Feature Contributions for Sample {sample_index}', fontsize=16)
plt.xticks(range(len(sample_SPE_contributions)), rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('SPE_Feature_Contributions.png', dpi=600)
plt.show()

# 绘制特征贡献度条形图（T2）
plt.figure(figsize=(10, 6))
plt.bar(range(len(original_feature_contributions_T2)), original_feature_contributions_T2, color='#1f77b4')
plt.xlabel('Feature Index', fontsize=14)
plt.ylabel('T2 Contribution', fontsize=14)
plt.title(f'T2 Feature Contributions for Sample {sample_index}', fontsize=16)
plt.xticks(range(len(original_feature_contributions_T2)), rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('T2_Feature_Contributions.png', dpi=600)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from sklearn.ensemble import IsolationForest
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 加载 PCA 模型
with open('pca_model.pkl', 'rb') as file:
    pca_loaded = pickle.load(file)
# 加载均值、标准差、两个阈值
params = np.load('X_train_params.npz')
X_train_mean, X_train_std, SPE_threshold, T2_threshold = params['mean'], params['std'], params['SPE'], params['T2']
# 读取评估数据，选取相应的评估数据
X_estimate_file = pd.read_csv('13_Data_评估数据1.csv', low_memory=False).iloc[0:, 1:]
# 删除多余列
columns_to_drop = []
X_estimate_file.drop(columns=columns_to_drop, axis=1, inplace=True)
X_estimate = (X_estimate_file.values).astype('float32')
# 定义归一化函数
def normalize_data(X, X_mean, X_std):
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std
# 归一化测试数据
X_estimate_normal = normalize_data(X_estimate, X_train_mean, X_train_std)
# 对训练数据和测试数据进行PCA降维
X_estimate_pca = pca_loaded.transform(X_estimate_normal )
# 计算训练数据和测试数据的逆PCA
X_estimate_reconstructed = pca_loaded.inverse_transform(X_estimate_pca)
# 计算训练数据和测试数据的SPE值
estimate_SPE = np.sum((X_estimate_normal - X_estimate_reconstructed )**2, axis=1)
# 使用SPE值T2值和阈值来预测异常
predictions_SPE = np.where(estimate_SPE > SPE_threshold, -1, 1)
num_samples = predictions_SPE.shape[0]
result = pd.DataFrame({'Time': range(num_samples), 'Lables' :predictions_SPE})
result.to_csv('predictions1.csv', index=False)
print('done')

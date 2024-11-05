import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# 定义归一化函数
def normalize_data(X, X_mean, X_std):
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std


# 加载 PCA 模型和训练集参数
with open('pca_model.pkl', 'rb') as file:
    pca_loaded = pickle.load(file)

params = np.load('X_train_params.npz')
X_train_mean, X_train_std, SPE_threshold, T2_threshold = params['mean'], params['std'], params['SPE'], params['T2']

# 数据文件和输出文件列表
data_files = [
    '13_Data_评估数据1.csv',
    '14_Data_评估数据2.csv',
    '15_Data_评估数据3.csv',
    '16_Data_评估数据4.csv',
    '17_Data_评估数据5.csv'
]
output_files = [
    'predictions1.csv',
    'predictions2.csv',
    'predictions3.csv',
    'predictions4.csv',
    'predictions5.csv'
]

# 设置一个用于填充缺失值的imputer
imputer = SimpleImputer(strategy='mean')

# 处理每个数据文件
for data_file, output_file in zip(data_files, output_files):
    # 读取评估数据
    X_estimate_file = pd.read_csv(data_file, low_memory=False).iloc[0:, 1:]
    X_estimate = (X_estimate_file.values).astype('float32')

    # 使用imputer填充NaN值
    X_estimate = imputer.fit_transform(X_estimate)

    # 归一化测试数据
    X_estimate_normal = normalize_data(X_estimate, X_train_mean, X_train_std)

    # 对训练数据和测试数据进行PCA降维
    X_estimate_pca = pca_loaded.transform(X_estimate_normal)

    # 计算训练数据和测试数据的逆PCA
    X_estimate_reconstructed = pca_loaded.inverse_transform(X_estimate_pca)

    # 计算训练数据和测试数据的SPE值
    estimate_SPE = np.sum((X_estimate_normal - X_estimate_reconstructed) ** 2, axis=1)

    # 使用SPE值和阈值来预测异常
    predictions_SPE = np.where(estimate_SPE > SPE_threshold, -1, 1)

    # 结果存入DataFrame
    result = pd.DataFrame({'Time': range(len(predictions_SPE)), 'Labels': predictions_SPE})

    # 保存到CSV文件
    result.to_csv(output_file, index=False)
    print(f'{output_file} 已生成.')

print('所有文件处理完毕。')

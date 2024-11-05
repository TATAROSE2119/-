import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

# 加载 PCA 模型和训练集参数
with open('pca_model.pkl', 'rb') as file:
    pca_loaded = pickle.load(file)

params = np.load('X_train_params.npz')
X_train_mean, X_train_std, SPE_threshold, T2_threshold = params['mean'], params['std'], params['SPE'], params['T2']


# 定义一个函数来处理每个数据文件
def process_file(file_path, output_file):
    # 读取数据文件
    X_estimate_file = pd.read_csv(file_path, low_memory=False).iloc[0:, 1:]
    X_estimate = (X_estimate_file.values).astype('float32')
    X_estimate_normal = (X_estimate - X_train_mean) / np.where(X_train_std == 0, 1, X_train_std)

    # 处理NaN值
    imputer = SimpleImputer(strategy='mean')
    X_estimate_normal = imputer.fit_transform(X_estimate_normal)  # 用均值填充NaN

    X_estimate_pca = pca_loaded.transform(X_estimate_normal)
    X_estimate_reconstructed = pca_loaded.inverse_transform(X_estimate_pca)
    estimate_SPE = np.sum((X_estimate_normal - X_estimate_reconstructed) ** 2, axis=1)

    # 生成标签，使得-1表示故障状态，1表示正常状态
    labels = (estimate_SPE > SPE_threshold).astype(int)
    labels = np.where(labels == 1, -1, 1)  # 修改标签逻辑

    # 训练并应用贝叶斯模型
    model = GaussianNB()
    model.fit(X_estimate_pca, labels)
    predictions = model.predict(X_estimate_pca)

    # 保存预测结果到CSV文件
    results = pd.DataFrame({
        'Time': np.arange(len(predictions)),  # 时间索引
        'Labels': predictions  # 贝叶斯模型的预测结果
    })
    results.to_csv(output_file, index=False)
    print(f'预测结果已保存到 {output_file} 文件中。')


# 文件列表和输出文件列表
files = [
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

# 循环处理每个文件
for file_path, output_file in zip(files, output_files):
    process_file(file_path, output_file)

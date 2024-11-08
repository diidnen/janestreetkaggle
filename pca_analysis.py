import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class PCAAnalyzer:
    def __init__(self, n_components=0.95):
        """
        初始化PCA分析器
        n_components: 可以是具体的特征数量，或者是要保留的方差比例(0-1之间)
        """
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
    
    def fit_transform(self, df):
        """对数据进行PCA转换"""
        # 标准化
        scaled_data = self.scaler.fit_transform(df)
        
        # PCA转换
        transformed_data = self.pca.fit_transform(scaled_data)
        
        # 创建新的DataFrame
        columns = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
        transformed_df = pd.DataFrame(transformed_data, columns=columns)
        
        return transformed_df
    
    def plot_explained_variance(self, output_dir='analysis_results/'):
        """绘制解释方差比例图"""
        plt.figure(figsize=(10, 6))
        
        # 累积解释方差比
        cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        
        # 绘制曲线
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
                cumulative_variance_ratio, 
                'bo-')
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs Number of Components')
        plt.grid(True)
        
        # 添加95%的参考线
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        plt.legend()
        
        plt.savefig(f'{output_dir}pca_variance.png')
        plt.close()
    
    def get_feature_importance(self, feature_names):
        """获取原始特征的重要性"""
        # 获取每个原始特征对主成分的贡献
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.components_.shape[0])],
            index=feature_names
        )
        
        # 计算每个特征的总体重要性
        importance = np.abs(loadings).mean(axis=1)
        
        return importance.sort_values(ascending=False)

def main():
    # 加载处理后的数据
    df_train = pd.read_csv('processed_train.csv')
    
    # 初始化PCA分析器
    pca_analyzer = PCAAnalyzer(n_components=0.95)
    
    # 转换数据
    transformed_df = pca_analyzer.fit_transform(df_train)
    
    # 绘制解释方差图
    pca_analyzer.plot_explained_variance()
    
    # 获取特征重要性
    feature_importance = pca_analyzer.get_feature_importance(df_train.columns)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # 保存转换后的数据
    transformed_df.to_csv('pca_transformed.csv', index=False)
    
    # 打印信息
    print(f"\nOriginal features: {df_train.shape[1]}")
    print(f"PCA features: {transformed_df.shape[1]}")
    print(f"Explained variance ratio: {pca_analyzer.pca.explained_variance_ratio_.sum():.4f}")

if __name__ == "__main__":
    main() 
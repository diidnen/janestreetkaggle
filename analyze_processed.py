import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
# 导入together.py中的处理函数
from process_data import process_data

class ProcessedDataAnalyzer:
    def __init__(self):
        self.output_dir = 'analysis_results/'  # 创建输出目录
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def analyze_data(self, train_path='processed_train.csv', test_path='processed_test.csv'):
        """分析处理后的数据"""
        print("Loading processed data...")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        print("\n开始生成可视化...")
        self.plot_distributions(df_train, df_test)
        self.plot_correlations(df_train)
        self.plot_feature_boxplots(df_train)
        
        print("\n生成统计信息...")
        self.print_statistics(df_train, df_test)
    
    def plot_distributions(self, df_train, df_test):
        """绘制特征分布图"""
        plt.figure(figsize=(15, 10))
        plt.clf()  # 清除当前图形
        
        features = df_train.select_dtypes(include=[np.number]).columns[:5]
        
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 3, i)
            sns.kdeplot(data=df_train[feature], label='Train', alpha=0.5)
            sns.kdeplot(data=df_test[feature], label='Test', alpha=0.5)
            plt.title(f'{feature} Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_distributions.png')
        plt.close('all')  # 确保关闭所有图形
        print(f"保存分布图到: {self.output_dir}feature_distributions.png")
    
    def plot_correlations(self, df):
        """绘制相关性热图"""
        plt.figure(figsize=(12, 8))
        plt.clf()
        
        corr = df.corr()
        top_features = corr.abs().mean().sort_values(ascending=False)[:15].index
        
        sns.heatmap(corr.loc[top_features, top_features], 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        
        plt.title('Feature Correlations (Top 15 Features)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}correlation_matrix.png')
        plt.close('all')
        print(f"保存相关性热图到: {self.output_dir}correlation_matrix.png")
    
    def plot_feature_boxplots(self, df):
        """绘制箱线图"""
        plt.figure(figsize=(15, 6))
        plt.clf()
        
        features = df.select_dtypes(include=[np.number]).columns[:10]
        
        sns.boxplot(data=df[features])
        plt.xticks(rotation=45)
        plt.title('Feature Distributions (Boxplots)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_boxplots.png')
        plt.close('all')
        print(f"保存箱线图到: {self.output_dir}feature_boxplots.png")
    
    def print_statistics(self, df_train, df_test):
        """打印基本统计信息"""
        print("\n=== 数据基本统计 ===")
        print(f"\n训练集形状: {df_train.shape}")
        print(f"测试集形状: {df_test.shape}")
        
        print("\n训练集描述性统计：")
        print(df_train.describe())
        
        # 检查缺失值
        missing_train = df_train.isnull().sum()
        missing_test = df_test.isnull().sum()
        
        if missing_train.any():
            print("\n训练集中的缺失值：")
            print(missing_train[missing_train > 0])
        
        if missing_test.any():
            print("\n测试集中的缺失值：")
            print(missing_test[missing_test > 0])

def main():
    print("开始数据处理和分析...")
    
    # 1. 加载原始数据
    print("加载原始数据...")
    df_train = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_500_10000.csv")
    df_test = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_10000_12000.csv")
    
    # 2. 使用together.py中的process_data处理数据
    print("处理数据...")
    X_train, y_train = process_data(df_train)
    X_test, y_test = process_data(df_test)
    
    # 3. 保存处理后的数据
    print("保存处理后的数据...")
    X_train.to_csv('processed_train.csv', index=False)
    X_test.to_csv('processed_test.csv', index=False)
    
    # 4. 分析处理后的数据
    print("分析处理后的数据...")
    analyzer = ProcessedDataAnalyzer()
    analyzer.analyze_data(
        train_path='processed_train.csv',
        test_path='processed_test.csv'
    )
    
    print("\n分析完成！请查看 analysis_results 文件夹中的可视化结果。")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main() 
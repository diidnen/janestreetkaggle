import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def process_data(df):
    """数据预处理主函数"""
    # 1. 删除完全缺失的特征
    cols_to_drop = ['feature_00', 'feature_01', 'feature_02', 'feature_03', 'feature_04',
                    'feature_21', 'feature_31', 'feature_26', 'feature_27']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. 处理目标变量
    if 'responder_6' in df.columns:
        y = (df['responder_6'] > 0).astype(int)
        df = df.drop('responder_6', axis=1)
    else:
        y = None
    
    # 3. 填充缺失值
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # 4. 创建新特征
    df = create_feature_combinations(df)
    
    # 5. 标准化
    scaler = RobustScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df, y

def create_feature_combinations(df):
    """创建特征组合"""
    features = [col for col in df.columns if 'feature_' in col]
    
    # 基础统计特征
    df['mean_features'] = df[features].mean(axis=1)
    df['std_features'] = df[features].std(axis=1)
    df['max_features'] = df[features].max(axis=1)
    df['min_features'] = df[features].min(axis=1)
    
    # 选择重要特征进行组合
    important_features = ['feature_48', 'feature_59', 'feature_11']
    for i in range(len(important_features)):
        for j in range(i+1, len(important_features)):
            f1, f2 = important_features[i], important_features[j]
            df[f'{f1}_{f2}_sum'] = df[f1] + df[f2]
            df[f'{f1}_{f2}_diff'] = df[f1] - df[f2]
            df[f'{f1}_{f2}_mult'] = df[f1] * df[f2]
    
    return df

def main():
    # 加载数据
    print("Loading data...")
    df_train = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_500_10000.csv")
    df_test = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_10000_12000.csv")
    
    # 处理数据
    print("Processing training data...")
    X_train, y_train = process_data(df_train)
    print("Processing test data...")
    X_test, y_test = process_data(df_test)
    
    # 保存处理后的数据
    print("Saving processed data...")
    X_train.to_csv('processed_train.csv', index=False)
    X_test.to_csv('processed_test.csv', index=False)
    
    # 打印处理结果
    print("\nProcessing complete!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print("\nNew features created:", [col for col in X_train.columns if '_' in col])
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main() 
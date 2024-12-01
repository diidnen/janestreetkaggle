import pandas as pd
import xgboost as xgb
import numpy as np

def main():
    # 加载数据
    print("Loading data...")
    test_data = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_10000_12000.csv")
    train_data = pd.read_csv("processed_train.csv")
    
    # 从训练数据中移除不需要的列
    X_train = train_data.drop(['Unnamed: 0', 'date_id', 'responder_6'], axis=1, errors='ignore')
    
    # 保持responder_6为连续值
    y_train = test_data['responder_6']
    y_test = y_train  # 在这个例子中我们用同一个数据集测试
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # 创建和训练模型 - 使用XGBRegressor而不是XGBClassifier
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        objective='reg:squarederror'  # 使用回归目标函数
    )
    
    # 确保X_train和y_train的样本数量匹配
    X_train = X_train.iloc[:len(y_train)]
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 获取预测值（现在是连续值）
    print("\nMaking predictions...")
    y_pred = model.predict(X_train)
    
    # 计算加权R²分数
    def weighted_r2(y_true, y_pred, weights):
        numerator = np.sum(weights * (y_true - y_pred) ** 2)
        denominator = np.sum(weights * y_true ** 2)
        r2 = 1 - numerator / denominator
        return r2
    
    # 假设weights是你的样本权重
    weights = np.ones(len(y_test))  # 如果有实际的权重，替换这行
    r2_score = weighted_r2(y_test, y_pred, weights)
    print(f"\nWeighted R² Score: {r2_score}")

if __name__ == "__main__":
    main() 

import pandas as pd
import xgboost as xgb
from evaluate_model import ModelEvaluator

def main():
    # 加载数据
    print("Loading data...")
    test_data = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_10000_12000.csv")
    train_data = pd.read_csv("processed_train.csv")
    
    # 从训练数据中移除标签列和不需要的列
    X_train = train_data.drop(['Unnamed: 0', 'date_id', 'responder_6'], axis=1, errors='ignore')
    
    # 准备标签
    y_train = test_data['responder_6']
    y_test = y_train  # 在这个例子中我们用同一个数据集测试
    
    # 转换为二分类问题
    y_train = (y_train > 0).astype(int)
    y_test = (y_test > 0).astype(int)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # 创建和训练模型
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42
    )
    
    # 确保X_train和y_train的样本数量匹配
    X_train = X_train.iloc[:len(y_train)]
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 获取预测概率
    print("\nMaking predictions...")
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    # 使用已有的ModelEvaluator进行评估
    print("\nGenerating evaluation plots...")
    evaluator = ModelEvaluator('analysis_results/')
    evaluator.plot_prediction_distribution(y_pred_proba)
    evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
    evaluator.plot_roc_curve(y_test, y_pred_proba)
    
    print("\nEvaluation complete! Check analysis_results folder for plots.")

if __name__ == "__main__":
    main() 
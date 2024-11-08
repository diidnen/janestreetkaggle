import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from evaluate_model import ModelEvaluator

class StackingEnsemble:
    def __init__(self):
        # 定义基础模型
        self.base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
            ('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
        ]
        # 定义元模型
        self.meta_model = LogisticRegression()
        
    def fit(self, X, y):
        # 生成基础模型的预测
        self.base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"Training {name}...")
            # 使用交叉验证生成预测
            self.base_predictions[:, i] = cross_val_predict(
                model, X, y, cv=5, method='predict_proba'
            )[:, 1]
            # 在全部数据上训练模型
            model.fit(X, y)
            
        # 训练元模型
        print("Training meta model...")
        self.meta_model.fit(self.base_predictions, y)
        return self
        
    def predict_proba(self, X):
        # 生成基础模型的预测
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = model.predict_proba(X)[:, 1]
            
        # 使用元模型进行最终预测
        return self.meta_model.predict_proba(meta_features)

def main():
    # 加载数据
    print("Loading data...")
    train_data = pd.read_csv("processed_train.csv")
    test_data = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_first_500.csv")
    
    # 准备特征和标签
    X = train_data.drop(['Unnamed: 0', 'date_id', 'responder_6'], axis=1, errors='ignore')
    y = (test_data['responder_6'] > 0).astype(int)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.iloc[:len(y)], y, test_size=0.2, random_state=42
    )
    
    # 训练集成模型
    ensemble = StackingEnsemble()
    ensemble.fit(X_train, y_train)
    
    # 预测
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # 评估
    evaluator = ModelEvaluator('analysis_results/')
    evaluator.plot_prediction_distribution(y_pred_proba)
    evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
    evaluator.plot_roc_curve(y_test, y_pred_proba)

if __name__ == "__main__":
    main() 
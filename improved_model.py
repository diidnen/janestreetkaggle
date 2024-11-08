import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

class ImprovedModel:
    def __init__(self):
        """初始化模型"""
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()
        self.model = None
        
    def process_data(self, df):
        """处理数据"""
        # 删除完全缺失的列
        df = df.dropna(axis=1, how='all')
        
        # 填充缺失值
        df = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        
        # 创建统计特征
        df['mean_features'] = df.mean(axis=1)
        df['std_features'] = df.std(axis=1)
        df['max_features'] = df.max(axis=1)
        df['min_features'] = df.min(axis=1)
        
        # 创建特征组合
        if 'feature_48' in df.columns and 'feature_59' in df.columns:
            df['f48_f59_sum'] = df['feature_48'] + df['feature_59']
            df['f48_f59_diff'] = df['feature_48'] - df['feature_59']
            df['f48_f59_mult'] = df['feature_48'] * df['feature_59']
        
        if 'feature_11' in df.columns:
            df['f11_squared'] = df['feature_11'] ** 2
            df['f11_cubed'] = df['feature_11'] ** 3
        
        # 标准化特征
        df = pd.DataFrame(
            self.scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return df
    
    def fit(self, X, y):
        """训练模型"""
        # 处理数据
        X = self.process_data(X)
        
        # 创建和训练模型
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        
        # 训练模型
        self.model.fit(
            X, 
            y,
            eval_metric=['auc', 'error'],
            verbose=True
        )
        
        return self
    
    def predict(self, X):
        """预测"""
        X = self.process_data(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        X = self.process_data(X)
        return self.model.predict_proba(X) 
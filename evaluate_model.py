import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from improved_model import ImprovedModel

class ModelEvaluator:
    def __init__(self, output_dir='analysis_results/'):
        self.output_dir = output_dir
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_prediction_distribution(self, y_pred_proba):
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba, bins=50, density=True)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.grid(True)
        plt.savefig(f'{self.output_dir}prediction_distribution.png')
        plt.close()
        
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.output_dir}precision_recall_curve.png')
        plt.close()
        
    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.output_dir}roc_curve.png')
        plt.close()

def main():
    # 加载数据
    print("Loading data...")
    df_train = pd.read_csv("processed_train.csv")
    df_test = pd.read_csv("processed_test.csv")
    
    # 加载标签
    y_train = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_first_500.csv")['responder_6']
    y_test = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_500_1000.csv")['responder_6']
    
    # 转换为二分类问题
    y_train = (y_train > 0).astype(int)
    y_test = (y_test > 0).astype(int)
    
    # 创建和训练模型
    print("Training model...")
    model = ImprovedModel()
    model.fit(df_train, y_train)
    
    # 获取预测概率
    print("Making predictions...")
    y_pred_proba = model.predict_proba(df_test)[:, 1]
    
    # 创建评估器
    print("Generating evaluation plots...")
    evaluator = ModelEvaluator()
    
    # 生成评估图
    evaluator.plot_prediction_distribution(y_pred_proba)
    print("- Saved prediction distribution plot")
    
    evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
    print("- Saved precision-recall curve")
    
    evaluator.plot_roc_curve(y_test, y_pred_proba)
    print("- Saved ROC curve")
    
    print("\nEvaluation complete! Check the analysis_results folder for plots.")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from evaluate_model import ModelEvaluator

def get_direct_predictions(test_data, threshold=0.5):
    """
    直接基于某些规则生成预测，不进行模型训练
    """
    # 这里可以根据你的业务规则直接生成预测概率
    # 例如：基于某些特征的简单阈值或规则
    
    # 生成一些示例预测概率
    y_pred_proba = np.random.uniform(0, 1, size=len(test_data))
    
    # 转换为二分类结果
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred_proba, y_pred

def main():
    # 加载测试数据
    print("Loading test data...")
    test_data = pd.read_csv("C:/Users/Devoir/PycharmProjects/pythonProject1/output_first_500.csv")
    
    # 直接生成预测
    y_pred_proba, y_pred = get_direct_predictions(test_data)
    
    # 评估结果
    evaluator = ModelEvaluator('analysis_results/')
    evaluator.plot_prediction_distribution(y_pred_proba)
    
    # 打印预测统计
    print(f"\n预测结果统计:")
    print(f"正类预测数量: {sum(y_pred == 1)}")
    print(f"负类预测数量: {sum(y_pred == 0)}")
    print(f"正类预测比例: {sum(y_pred == 1) / len(y_pred):.2%}")

if __name__ == "__main__":
    main() 
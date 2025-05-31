# 假设我们保存以下代码为 get_my_auc.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# 简化版本，只计算AUC
def get_my_auc(predictions_file, gt_file):
    # 读取真实标签
    gts = pd.read_excel(gt_file)["labels"]
    gts = np.array(gts)

    # 读取您的预测
    preds = pd.read_excel(predictions_file, sheet_name="predictions")["predictions"]
    preds = np.array(preds)

    # 计算AUC
    auc = roc_auc_score(gts, preds)
    print(f"您的AUC分数是: {auc:.4f}")
    return auc


# 使用示例
if __name__ == "__main__":
    gt_file = "./result/gts.xlsx"
    your_predictions = "./result/安全对抗小组.xlsx"
    get_my_auc(your_predictions, gt_file)
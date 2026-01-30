"""
Evaluation Metrics
对应论文中的 Accuracy, Precision, Recall, F1-score, AUC
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true, y_pred, y_prob=None, num_classes=5, average='weighted'):
    """
    计算论文中使用的所有评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率 (用于AUC计算)
        num_classes: 类别数
        average: 多分类平均方法 ('weighted', 'macro', 'micro')
    """
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUC (One-vs-Rest)
    if y_prob is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            metrics['auc'] = roc_auc_score(y_true_bin, y_prob, average=average, multi_class='ovr')
        except ValueError:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    打印详细分类报告
    target_names: ['SSTR', 'THRO', 'WSTR', 'SILIS', 'MICR']
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


def calculate_top_k_accuracy(y_true, y_prob, k=5):
    """
    计算 Top-K 准确率 (虽然论文主要用 Top-1，但保留此功能)
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
    return np.mean(correct)
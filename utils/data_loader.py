"""
Data Loading and Preprocessing
包括 SMOTE 数据增强 (Synthetic Minority Over-sampling Technique)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class WellLogDataset(Dataset):
    """
    测井数据集类
    处理8种测井曲线: AC, CAL, CNL, DEN, GR, PE, RLLD, RLLS
    """
    def __init__(self, data, labels, seq_len=10, stride=5):
        """
        data: numpy array [N, 8] - 8种测井曲线
        labels: numpy array [N] - 岩性标签
        seq_len: 序列长度（对应论文中的窗口大小）
        stride: 滑动步长
        """
        self.seq_len = seq_len
        self.stride = stride
        
        # 标准化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)
        
        # 创建序列数据
        self.sequences, self.seq_labels = self._create_sequences()
        
    def _create_sequences(self):
        """创建滑动窗口序列"""
        sequences = []
        seq_labels = []
        
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
            seq = self.data[i:i+self.seq_len]
            label = self.labels[i + self.seq_len // 2]  # 使用窗口中间点的标签
            
            sequences.append(seq)
            seq_labels.append(label)
            
        return np.array(sequences), np.array(seq_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.sequences[idx])
        y = torch.LongTensor([self.seq_labels[idx]])[0]
        return x, y
    
    def get_scaler(self):
        return self.scaler
    
    def get_label_encoder(self):
        return self.label_encoder


def load_and_preprocess_data(file_path, 
                            log_columns=['AC', 'CAL', 'CNL', 'DEN', 'GR', 'PE', 'RLLD', 'RLLS'],
                            label_column='Lithology',
                            use_smote=True,
                            test_size=0.2,
                            random_state=42):
    """
    加载数据并进行预处理，包括SMOTE数据增强
    
    返回: train_loader, val_loader, test_loader, scaler, label_encoder
    """
    # 读取CSV
    df = pd.read_csv(file_path)
    
    # 提取测井数据和标签
    X = df[log_columns].values
    y = df[label_column].values
    
    # 划分训练集和测试集 (论文中是 60% train, 30% val, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )
    
    # SMOTE 数据增强 (处理类别不平衡，如图5所示)
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_train)} samples")
    
    # 创建 Dataset
    train_dataset = WellLogDataset(X_train, y_train)
    val_dataset = WellLogDataset(X_val, y_val)
    test_dataset = WellLogDataset(X_test, y_test)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return (train_loader, val_loader, test_loader, 
            train_dataset.get_scaler(), 
            train_dataset.get_label_encoder())


def prepare_transfer_learning_data(source_file, target_file, log_columns, label_column):
    """
    准备迁移学习数据
    用于论文5.5节的微调实验 (Dengying-2 和 Leikoupo-43)
    """
    # 源域数据 (预训练)
    source_df = pd.read_csv(source_file)
    X_source = source_df[log_columns].values
    y_source = source_df[label_column].values
    
    # 目标域数据 (微调)
    target_df = pd.read_csv(target_file)
    X_target = target_df[log_columns].values
    y_target = target_df[label_column].values
    
    return (X_source, y_source), (X_target, y_target)
"""
Transfer Learning / Fine-tuning Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

from models.gcn import MicrobialiteGCN
from utils.data_loader import prepare_transfer_learning_data


class TransferLearningGCN:
    """
    迁移学习策略：
    1. 加载预训练GCN模型
    2. 替换/修改最后一层适应新类别数
    3. 可选择冻结前面层进行特征提取 + SVM分类
    """
    def __init__(self, pretrained_path, num_new_classes, freeze_layers=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练模型
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        self.model = MicrobialiteGCN(num_classes=5)  # 原5类
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 修改最后一层以适应新类别
        if num_new_classes != 5:
            in_features = self.model.fc2.in_features
            self.model.fc2 = nn.Linear(in_features, num_new_classes).to(self.device)
        
        self.model = self.model.to(self.device)
        
        if freeze_layers:
            # 冻结GRU和GCN层，只训练分类层
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # 非全连接层
                    param.requires_grad = False
        
    def fine_tune(self, X_target, y_target, epochs=50, lr=1e-4):
        """
        在目标域数据上微调
        """
        # 准备数据
        from utils.data_loader import WellLogDataset
        dataset = WellLogDataset(X_target, y_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits, _ = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
        
    def extract_features(self, X):
        """
        使用预训练GCN提取特征，用于SVM分类
        对应论文中 "GCN + SVM" 的策略(小样本情况下)
        """
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        features = []
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                # 获取倒数第二层的特征
                _, h_n = self.model.gru(X_batch)
                features.append(h_n[-1].cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def fine_tune_with_svm(self, X_train, y_train, X_test, y_test):
        """
        GCN特征提取 + SVM分类 (用于极少样本情况，如论文中的500样本)
        """
        print("Extracting features with pretrained GCN...")
        train_features = self.extract_features(X_train)
        test_features = self.extract_features(X_test)
        
        print("Training SVM classifier...")
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(train_features, y_train)
        
        y_pred = svm.predict(test_features)
        print(classification_report(y_test, y_pred))
        
        return svm


if __name__ == '__main__':
    # 示例：在Dengying-2地层上进行微调
    source_file = 'data/dengying4.csv'
    target_file = 'data/dengying2.csv'
    
    log_cols = ['AC', 'CAL', 'CNL', 'DEN', 'GR', 'PE', 'RLLD', 'RLLS']
    
    (X_src, y_src), (X_tgt, y_tgt) = prepare_transfer_learning_data(
        source_file, target_file, log_cols, 'Lithology'
    )
    
    # 划分目标域训练测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tgt, y_tgt, test_size=0.2, random_state=42
    )
    
    # 迁移学习
    tl = TransferLearningGCN('checkpoints/best_gcn.pth', 
                            num_new_classes=6,  # Dengying-2有6类
                            freeze_layers=True)
    
    if len(X_train) < 1000:
        # 小样本：使用GCN+SVM
        tl.fine_tune_with_svm(X_train, y_train, X_test, y_test)
    else:
        # 正常微调
        tl.fine_tune(X_train, y_train, epochs=50)
        # 评估...
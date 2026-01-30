"""
Training Script for GCN and Baseline Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
from tqdm import tqdm

from models.gcn import MicrobialiteGCN, MicrobialiteGCNv2
from models.baselines import LSTMClassifier, RNNClassifier, TCNClassifier, ANNClassifier, DropoutANN
from utils.data_loader import load_and_preprocess_data
from utils.metrics import calculate_metrics


def get_model(model_name, input_dim, num_classes, config):
    """根据模型名称获取对应模型"""
    if model_name == 'gcn':
        return MicrobialiteGCN(input_dim, num_classes=num_classes, 
                              hidden_dim=config['hidden_dim'],
                              window_size=config['window_size'])
    elif model_name == 'gcn_v2':
        return MicrobialiteGCNv2(input_dim, num_classes=num_classes,
                                hidden_dim=config['hidden_dim'])
    elif model_name == 'lstm':
        return LSTMClassifier(input_dim, num_classes=num_classes,
                             hidden_dim=config['hidden_dim'],
                             num_layers=config.get('num_layers', 5))
    elif model_name == 'rnn':
        return RNNClassifier(input_dim, num_classes=num_classes,
                            hidden_dim=config['hidden_dim'],
                            num_layers=config.get('num_layers', 5))
    elif model_name == 'tcn':
        return TCNClassifier(input_dim, num_classes=num_classes)
    elif model_name == 'ann':
        return ANNClassifier(input_dim, seq_len=config['seq_len'], 
                            num_classes=num_classes)
    elif model_name == 'dropout_ann':
        return DropoutANN(input_dim, seq_len=config['seq_len'], 
                         num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for X, y in tqdm(dataloader, desc='Training'):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # GCN返回 logits 和 adj_matrix，基线模型只返回 logits
        output = model(X)
        if isinstance(output, tuple):
            logits, adj = output
        else:
            logits = output
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录预测
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Evaluating'):
            X, y = X.to(device), y.to(device)
            
            output = model(X)
            if isinstance(output, tuple):
                logits, adj = output
            else:
                logits = output
            
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, default='gcn', 
                       choices=['gcn', 'gcn_v2', 'lstm', 'rnn', 'tcn', 'ann', 'dropout_ann'])
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    log_columns = config['log_columns']  # ['AC', 'CAL', 'CNL', 'DEN', 'GR', 'PE', 'RLLD', 'RLLS']
    train_loader, val_loader, test_loader, scaler, label_encoder = load_and_preprocess_data(
        config['data_path'],
        log_columns=log_columns,
        use_smote=config.get('use_smote', True)
    )
    
    # 模型
    input_dim = len(log_columns)
    num_classes = config['num_classes']  # 5
    
    model = get_model(args.model, input_dim, num_classes, config).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # TensorBoard
    writer = SummaryWriter(f'runs/{args.model}_{config["exp_name"]}')
    
    # 训练循环
    best_val_auc = 0
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        # 记录
        writer.add_scalars('Loss', {'train': train_metrics['loss'], 
                                   'val': val_metrics['loss']}, epoch)
        writer.add_scalars('Accuracy', {'train': train_metrics['accuracy'], 
                                       'val': val_metrics['accuracy']}, epoch)
        writer.add_scalars('F1', {'train': train_metrics['f1'], 
                                 'val': val_metrics['f1']}, epoch)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"AUC: {train_metrics.get('auc', 0):.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics.get('auc', 0):.4f}")
        
        # 保存最佳模型
        if val_metrics.get('auc', 0) > best_val_auc:
            best_val_auc = val_metrics['auc']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'scaler': scaler,
                'label_encoder': label_encoder
            }, f'checkpoints/best_{args.model}.pth')
    
    # 最终测试
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics.get('auc', 0):.4f}")
    
    writer.close()


if __name__ == '__main__':
    main()
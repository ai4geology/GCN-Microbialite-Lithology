"""
Core layers for GCN Lithology Identification
包括：GLU, Graph Fourier Transform, Graph Convolution, Attention Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eig


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) for GCN
    Formula: (X * W + b) ⊗ (X * V + c)
    """
    def __init__(self, in_features, out_features):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # x: [batch, seq, features]
        gate = torch.sigmoid(self.linear2(x))
        return self.linear1(x) * gate


class GraphFourierTransform:
    """
    图傅里叶变换 (GFT)
    基于拉普拉斯矩阵的特征分解
    """
    @staticmethod
    def compute_laplacian(adj_matrix):
        """
        计算对称归一化拉普拉斯矩阵 L_sym = D^(-1/2) * (D - A) * D^(-1/2)
        adj_matrix: 邻接矩阵 [N, N]
        """
        # 度矩阵 D
        degree = adj_matrix.sum(dim=-1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        
        # L = I - D^(-1/2) * A * D^(-1/2)
        L = torch.eye(adj_matrix.size(0)).to(adj_matrix.device) - \
            D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        return L
    
    @staticmethod
    def transform(x, laplacian):
        """
        图傅里叶变换: x_hat = U^T * x
        其中 U 是拉普拉斯矩阵的特征向量
        x: [N, features] 或 [batch, N, features]
        """
        # 特征分解 L = U * Lambda * U^T
        L_np = laplacian.cpu().numpy()
        eigenvalues, eigenvectors = eig(L_np)
        
        # 按特征值排序
        idx = eigenvalues.argsort()
        eigenvectors = eigenvectors[:, idx]
        
        U = torch.from_numpy(eigenvectors).float().to(laplacian.device)
        
        if x.dim() == 2:
            return U.T @ x, U
        else:
            # batch mode
            return torch.bmm(U.T.unsqueeze(0).expand(x.size(0), -1, -1), x), U


class SpectralGraphConv(nn.Module):
    """
    频谱图卷积层
    包含 DFT -> Conv -> GLU -> iDFT 流程
    """
    def __init__(self, in_features, out_features, num_logs=8):
        super(SpectralGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_logs = num_logs
        
        # 1D 卷积用于频域处理
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=3, padding=1)
        
        # GLU 门控
        self.glu = GLU(out_features, out_features)
        
        # 学习邻接矩阵的参数
        self.attention = nn.MultiheadAttention(in_features, num_heads=4, batch_first=True)
        
    def forward(self, x, adj_matrix):
        """
        x: [batch, seq_len, num_logs, features]
        adj_matrix: [batch, num_logs, num_logs] 邻接矩阵
        """
        batch_size, seq_len, num_logs, features = x.shape
        
        outputs = []
        for t in range(seq_len):
            # 当前时间步的图 [batch, num_logs, features]
            x_t = x[:, t, :, :]
            
            # 计算拉普拉斯矩阵并进行图傅里叶变换
            laplacian = GraphFourierTransform.compute_laplacian(adj_matrix)
            
            # DFT 近似使用 FFT
            x_freq = torch.fft.fft(x_t, dim=-1).real
            
            # 卷积处理
            x_conv = self.conv(x_freq.transpose(1, 2)).transpose(1, 2)
            
            # GLU
            x_glu = self.glu(x_conv)
            
            # iDFT
            x_time = torch.fft.ifft(x_glu, dim=-1).real
            
            outputs.append(x_time.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)  # [batch, seq, num_logs, features]


class SelfAttentionAdj(nn.Module):
    """
    自注意力机制生成邻接矩阵
    论文中用于从GRU隐藏状态生成邻接矩阵
    """
    def __init__(self, hidden_dim):
        super(SelfAttentionAdj, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hidden_states):
        """
        hidden_states: [batch, seq, num_logs, hidden]
        返回邻接矩阵 [batch, num_logs, num_logs]
        """
        # 取最后一个时间步或平均
        h = hidden_states.mean(dim=1)  # [batch, num_logs, hidden]
        
        Q = self.query(h)
        K = self.key(h)
        
        # 注意力分数作为邻接矩阵
        attn = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(Q.size(-1))
        adj = F.softmax(attn, dim=-1)
        
        return adj
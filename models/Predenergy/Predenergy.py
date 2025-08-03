import math
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Type, Dict, Optional, Tuple, Union, List
from torch import optim
import os
import json
from einops import rearrange

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.data_processing import split_before
from models.ABS_Predenergy_model import PredenergyMoTSEConfig, PredenergyForPrediction
from models.model_base import ModelBase, BatchMaker

# 导入数据处理模块
from datasets import (
    PredenergyDataset, 
    PredenergyWindowDataset, 
    PredenergyDataLoader,
    create_Predenergy_data_loader
)

# 导入训练相关模块
from runner import PredenergyRunner, setup_seed
from trainer.hf_trainer import PredenergyTrainer, PredenergyTrainingArguments

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class Predenergy(ModelBase):
   
    def __init__(self, **kwargs):
        super(Predenergy, self).__init__()
        # Check if we should use the combined model
        self.use_combined_model = kwargs.get('use_combined_model', False)
        
        if self.use_combined_model:
            self.config = PredenergyMoTSEConfig(**kwargs)
        else:
            self.config = TransformerConfig(**kwargs)
        
        # 验证配置
        self._validate_initial_config()
            
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        
        # 数据处理相关属性
        self.data_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 模型相关属性
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
        
        # 训练相关属性
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        print(f"Predenergy model initialized with config:")
        print(f"  - Use combined model: {self.use_combined_model}")
        print(f"  - Sequence length: {self.seq_len}")
        print(f"  - Horizon: {getattr(self.config, 'horizon', 'N/A')}")
        print(f"  - Device: {self.device}")

    @property
    def model_name(self):
        return "Predenergy"

    @staticmethod
    def required_hyper_params() -> dict:
        
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm"
        }

    def _validate_initial_config(self):
        """验证初始配置"""
        required_params = ['seq_len', 'horizon']
        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing required parameter: {param}")
        
        if self.config.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        
        if hasattr(self.config, 'horizon') and self.config.horizon <= 0:
            raise ValueError("horizon must be positive")
        
        if hasattr(self.config, 'batch_size') and self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if hasattr(self.config, 'lr') and self.config.lr <= 0:
            raise ValueError("learning rate must be positive")
        
        print("Configuration validation passed")

    def __repr__(self) -> str:
        return f"Predenergy(seq_len={self.seq_len}, horizon={self.config.horizon}, use_combined_model={self.use_combined_model})"

    def setup_data_loader(self, data: Union[str, np.ndarray, pd.DataFrame], **kwargs):
        
        try:
            loader_type = kwargs.get('loader_type', 'standard')
            
            # 验证数据
            if data is None:
                raise ValueError("Data cannot be None")
            
            if isinstance(data, pd.DataFrame) and data.empty:
                raise ValueError("DataFrame is empty")
            
            if isinstance(data, np.ndarray) and data.size == 0:
                raise ValueError("NumPy array is empty")
            
            # 创建数据加载器
            self.data_loader = create_Predenergy_data_loader(
                data=data,
                loader_type=loader_type,
                seq_len=self.seq_len,
                pred_len=self.config.horizon,
                batch_size=kwargs.get('batch_size', self.config.batch_size),
                features=kwargs.get('features', 'S'),
                target=kwargs.get('target', 'OT'),
                timeenc=kwargs.get('timeenc', 0),
                freq=kwargs.get('freq', self.config.freq),
                normalize=kwargs.get('normalize', 2),
                train_ratio=kwargs.get('train_ratio', 0.7),
                val_ratio=kwargs.get('val_ratio', 0.2),
                shuffle=kwargs.get('shuffle', True),
                num_workers=kwargs.get('num_workers', self.config.num_workers),
                **kwargs
            )
            
            # 获取数据加载器
            self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_all_loaders()
            
            print(f"Data loader created successfully")
            print(f"Train samples: {len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 'Unknown'}")
            print(f"Val samples: {len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 'Unknown'}")
            print(f"Test samples: {len(self.test_loader.dataset) if hasattr(self.test_loader, 'dataset') else 'Unknown'}")
            
            return self.data_loader
        except Exception as e:
            print(f"Error setting up data loader: {e}")
            raise

    def _create_model(self):
        """创建模型"""
        try:
            if self.use_combined_model:
                self.model = PredenergyForPrediction(self.config)
            else:
                # 创建原始Predenergy模型
                from models.modeling_Predenergy import PredenergyForPrediction, PredenergyConfig
                config = PredenergyConfig(
                    seq_len=self.config.seq_len,
                    pred_len=self.config.horizon,
                    enc_in=self.config.enc_in,
                    dec_in=self.config.dec_in,
                    c_out=self.config.c_out,
                    d_model=self.config.d_model,
                    n_heads=self.config.n_heads,
                    e_layers=self.config.e_layers,
                    d_layers=self.config.d_layers,
                    d_ff=self.config.d_ff,
                    dropout=self.config.dropout,
                    activation=self.config.activation,
                    output_attention=self.config.output_attention,
                    freq=self.config.freq,
                    embed='timeF',
                    distil=True,
                    factor=self.config.factor,
                    moving_avg=self.config.moving_avg,
                    patch_len=self.config.patch_len,
                    stride=self.config.stride,
                    period_len=self.config.period_len,
                    label_len=self.config.seq_len // 2,
                )
                self.model = PredenergyForPrediction(config)
            
            self.model.to(self.device)
            print(f"Model created successfully on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            return self.model
        except Exception as e:
            print(f"Error creating model: {e}")
            raise

    def _setup_training_components(self):

        # 损失函数
        if self.config.loss == 'huber':
            self.criterion = torch.nn.HuberLoss(reduction='none', delta=2.0)
        elif self.config.loss == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='none')
        elif self.config.loss == 'mae':
            self.criterion = torch.nn.L1Loss(reduction='none')
        else:
            self.criterion = torch.nn.HuberLoss(reduction='none', delta=2.0)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            delta=1e-6
        )

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):

        if self.data_loader is None:
            self.setup_data_loader(train_data, features='M')
        
        # 获取数据集信息
        dataset_info = self.data_loader.get_dataset_info()
        print(f"Dataset info: {dataset_info}")
        
        pass

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):

        if self.data_loader is None:
            self.setup_data_loader(train_data, features='S')
        
        # 获取数据集信息
        dataset_info = self.data_loader.get_dataset_info()
        print(f"Dataset info: {dataset_info}")
        
        pass

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):

        if self.data_loader is None:
            self.setup_data_loader(train_data, loader_type='forecasting')
        
        # 获取数据集信息
        dataset_info = self.data_loader.get_dataset_info()
        print(f"Dataset info: {dataset_info}")
        
        pass

    def padding_data_for_forecast(self, test):

        if hasattr(self, 'data_loader') and self.data_loader is not None:
            # 使用已设置的数据加载器
            return test
        else:
            # 使用原有的填充方法
            return self._original_padding_method(test)

    def _original_padding_method(self, test):

        if len(test.shape) == 1:
            test = test.reshape(-1, 1)
        
        # 如果测试数据长度小于seq_len，进行填充
        if test.shape[0] < self.seq_len:
            padding_length = self.seq_len - test.shape[0]
            padding = np.zeros((padding_length, test.shape[1]))
            test = np.vstack([padding, test])
        
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:

        if padding_len > 0:
            padding = np.zeros((padding_len, time_stamps_list.shape[1]))
            return np.vstack([padding, time_stamps_list])
        return time_stamps_list

    def validate(self, valid_data_loader, criterion):

        if self.use_combined_model:
            # 使用组合模型的验证逻辑
            return self._validate_combined_model(valid_data_loader, criterion)
        else:
            # 使用原始Predenergy的验证逻辑
            return self._validate_original_model(valid_data_loader, criterion)

    def _validate_combined_model(self, valid_data_loader, criterion):

        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in valid_data_loader:
                # 处理批次数据格式
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    loss_masks = batch.get('loss_masks', None)
                    if loss_masks is not None:
                        loss_masks = loss_masks.to(self.device)
                else:
                    # 如果是元组或列表格式
                    if len(batch) >= 2:
                        inputs, labels = batch[0], batch[1]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        loss_masks = batch[2] if len(batch) > 2 else None
                        if loss_masks is not None:
                            loss_masks = loss_masks.to(self.device)
                    else:
                        inputs = batch[0].to(self.device)
                        labels = None
                        loss_masks = None
                
                # 确保输入格式正确
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(0)  # 添加批次维度
                
                outputs = self.model(
                    input=inputs,
                    labels=labels,
                    loss_masks=loss_masks
                )
                
                loss = outputs['loss']
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        return total_loss / total_samples

    def _validate_original_model(self, valid_data_loader, criterion):

        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in valid_data_loader:
                # 处理批次数据格式
                if isinstance(batch, dict):
                    x_enc = batch['x_enc'].to(self.device)
                    x_mark_enc = batch['x_mark_enc'].to(self.device)
                    x_dec = batch['x_dec'].to(self.device)
                    x_mark_dec = batch['x_mark_dec'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    # 如果是元组或列表格式
                    if len(batch) >= 5:
                        x_enc, x_mark_enc, x_dec, x_mark_dec, labels = batch
                        x_enc = x_enc.to(self.device)
                        x_mark_enc = x_mark_enc.to(self.device)
                        x_dec = x_dec.to(self.device)
                        x_mark_dec = x_mark_dec.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        # 简化格式处理
                        x_enc = batch[0].to(self.device)
                        x_mark_enc = torch.zeros_like(x_enc)
                        x_dec = torch.zeros_like(x_enc)
                        x_mark_dec = torch.zeros_like(x_enc)
                        labels = batch[1].to(self.device) if len(batch) > 1 else None
                
                outputs = self.model(
                    x_enc=x_enc,
                    x_mark_enc=x_mark_enc,
                    x_dec=x_dec,
                    x_mark_dec=x_mark_dec,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item() * x_enc.size(0)
                total_samples += x_enc.size(0)
        
        return total_loss / total_samples

    def forecast_fit(self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float) -> "ModelBase":

        # 设置数据加载器
        if self.data_loader is None:
            self.setup_data_loader(train_valid_data)
        
        # 创建模型
        if self.model is None:
            self._create_model()
        
        # 设置训练组件
        self._setup_training_components()
        
        # 开始训练
        self._train_model()
        
        self.is_fitted = True
        return self

    def _train_model(self):

        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.lr}")
        
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_samples = 0
            
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                # 处理批次数据格式
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    loss_masks = batch.get('loss_masks', None)
                    if loss_masks is not None:
                        loss_masks = loss_masks.to(self.device)
                else:
                    # 如果是元组或列表格式
                    if len(batch) >= 2:
                        inputs, labels = batch[0], batch[1]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        loss_masks = batch[2] if len(batch) > 2 else None
                        if loss_masks is not None:
                            loss_masks = loss_masks.to(self.device)
                    else:
                        inputs = batch[0].to(self.device)
                        labels = None
                        loss_masks = None
                
                # 确保输入格式正确
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(0)  # 添加批次维度
                
                outputs = self.model(
                    input=inputs,
                    labels=labels,
                    loss_masks=loss_masks
                )
                
                loss = outputs['loss']
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)
            
            # 验证阶段
            val_loss = self.validate(self.val_loader, self.criterion)
            
            # 学习率调度
            self.scheduler.step()
            
            # 早停检查
            improved = self.early_stopping(val_loss, self.model)
            
            if improved:
                best_val_loss = val_loss
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            # 记录训练历史
            avg_train_loss = train_loss / train_samples
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_loss)
            
            # 打印训练信息
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # 加载最佳模型
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
            os.remove('best_model.pth')
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        # 保存训练历史
        self.training_history = training_history

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if train is None or (isinstance(train, pd.DataFrame) and train.empty):
            raise ValueError("Training data cannot be None or empty")
        
        try:
            if self.use_combined_model:
                return self._forecast_combined_model(horizon, train)
            else:
                return self._forecast_original_model(horizon, train)
        except Exception as e:
            print(f"Error during forecasting: {e}")
            raise

    def _forecast_combined_model(self, horizon: int, train: pd.DataFrame) -> np.ndarray:

        self.model.eval()
        
        # 准备输入数据
        if isinstance(train, pd.DataFrame):
            input_data = train.values
        else:
            input_data = train
        
        # 数据预处理
        input_data = self._preprocess_input(input_data)
        
        # 转换为张量
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.predict(input_tensor, max_horizon_length=horizon)
            
            if isinstance(outputs, dict):
                predictions = outputs['predictions']
            else:
                predictions = outputs
            
            # 转换为numpy数组
            predictions = predictions.cpu().numpy()
            
            # 后处理
            predictions = self._postprocess_output(predictions)
            
            return predictions

    def _forecast_original_model(self, horizon: int, train: pd.DataFrame) -> np.ndarray:

        self.model.eval()
        
        # 准备输入数据
        if isinstance(train, pd.DataFrame):
            input_data = train.values
        else:
            input_data = train
        
        # 数据预处理
        input_data = self._preprocess_input(input_data)
        
        # 转换为张量
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 对于原始模型，需要准备编码器和解码器输入
            x_enc = input_tensor
            x_mark_enc = torch.zeros_like(x_enc)  # 时间特征
            x_dec = torch.zeros((1, horizon, input_tensor.shape[-1]), dtype=torch.float32).to(self.device)
            x_mark_dec = torch.zeros((1, horizon, input_tensor.shape[-1]), dtype=torch.float32).to(self.device)
            
            outputs = self.model.predict(
                x_enc=x_enc,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_mark_dec=x_mark_dec
            )
            
            # 转换为numpy数组
            predictions = outputs.cpu().numpy()
            
            # 后处理
            predictions = self._postprocess_output(predictions)
            
            return predictions

    def _preprocess_input(self, input_data: np.ndarray) -> np.ndarray:

        # 确保维度正确
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(-1, 1)
        
        # 标准化
        if hasattr(self, 'scaler') and self.scaler is not None:
            try:
                input_data = self.scaler.transform(input_data)
            except Exception as e:
                print(f"Warning: Error in scaling input data: {e}")
                # 如果标准化失败，使用原始数据
                pass
        
        # 如果输入长度小于seq_len，进行填充
        if input_data.shape[0] < self.seq_len:
            padding_length = self.seq_len - input_data.shape[0]
            padding = np.zeros((padding_length, input_data.shape[1]))
            input_data = np.vstack([padding, input_data])
        
        return input_data

    def _postprocess_output(self, predictions: np.ndarray) -> np.ndarray:

        # 反标准化
        if hasattr(self, 'scaler') and self.scaler is not None:
            try:
                predictions = self.scaler.inverse_transform(predictions)
            except Exception as e:
                print(f"Warning: Error in inverse scaling predictions: {e}")
                # 如果反标准化失败，返回原始预测
                pass
        
        return predictions

    def save_model(self, path: str):

        if self.model is not None:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'scaler': self.scaler,
                    'is_fitted': self.is_fitted
                }, path)
                print(f"Model saved successfully to {path}")
            except Exception as e:
                print(f"Error saving model: {e}")
                raise
        else:
            raise ValueError("No model to save")

    def load_model(self, path: str):

        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.config = checkpoint['config']
                self.scaler = checkpoint['scaler']
                self.is_fitted = checkpoint['is_fitted']
                
                # 创建模型
                self._create_model()
                
                # 加载模型权重
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print(f"Model loaded successfully from {path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    def get_model_info(self) -> Dict:

        try:
            return {
                'model_name': self.model_name,
                'use_combined_model': self.use_combined_model,
                'seq_len': self.seq_len,
                'horizon': self.config.horizon,
                'is_fitted': self.is_fitted,
                'device': str(self.device),
                'total_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
            }
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {'error': str(e)}

    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """评估模型性能"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before evaluation")
            
            if test_data is None or (isinstance(test_data, pd.DataFrame) and test_data.empty):
                raise ValueError("Test data cannot be None or empty")
            
            # 准备测试数据
            if self.test_loader is None:
                # 如果没有测试数据加载器，创建一个
                test_data_loader = self.setup_data_loader(test_data)
                _, _, test_loader = test_data_loader.get_all_loaders()
            else:
                test_loader = self.test_loader
            
            # 评估模型
            self.model.eval()
            total_loss = 0
            total_samples = 0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        if isinstance(batch, dict):
                            inputs = batch['input_ids'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            loss_masks = batch.get('loss_masks', None)
                            if loss_masks is not None:
                                loss_masks = loss_masks.to(self.device)
                        else:
                            inputs, labels = batch
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                            loss_masks = None
                        
                        outputs = self.model(
                            input=inputs,
                            labels=labels,
                            loss_masks=loss_masks
                        )
                        
                        loss = outputs['loss']
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)
                        
                        # 收集预测结果
                        if 'predictions' in outputs:
                            pred = outputs['predictions'].cpu().numpy()
                            predictions.extend(pred)
                        else:
                            # 如果没有直接输出预测，使用输入进行预测
                            pred = self.model.predict(inputs, max_horizon_length=labels.size(1))
                            if isinstance(pred, dict):
                                pred = pred['predictions']
                            pred = pred.cpu().numpy()
                            predictions.extend(pred)
                        
                        actuals.extend(labels.cpu().numpy())
                    except Exception as e:
                        print(f"Warning: Error processing batch: {e}")
                        continue
            
            if total_samples == 0:
                raise ValueError("No valid samples for evaluation")
            
            # 计算评估指标
            avg_loss = total_loss / total_samples
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # 计算各种评估指标
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(mse)
            
            # 计算MAPE
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
            
            # 计算SMAPE
            smape = 2.0 * np.mean(np.abs(predictions - actuals) / (np.abs(predictions) + np.abs(actuals) + 1e-8)) * 100
            
            return {
                'loss': avg_loss,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'smape': smape,
                'predictions': predictions,
                'actuals': actuals
            }
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return {'error': str(e)}

    def fit(self, train_data: pd.DataFrame, **kwargs) -> "Predenergy":
        return self.forecast_fit(train_data, train_ratio_in_tv=0.8)

    def predict(self, data: pd.DataFrame, horizon: int = None) -> np.ndarray:
        if horizon is None:
            horizon = self.config.horizon
        return self.forecast(horizon, data)

    def get_feature_importance(self) -> Dict:
        """获取特征重要性（如果模型支持）"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting feature importance")
            
            # 对于Transformer模型，特征重要性可能不太直接
            # 这里返回模型的基本信息
            return {
                'model_type': 'Predenergy',
                'use_combined_model': self.use_combined_model,
                'total_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0,
                'note': 'Feature importance not directly available for Transformer-based models'
            }
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {'error': str(e)}

    def get_model_summary(self) -> str:
        """获取模型摘要"""
        try:
            if self.model is None:
                return "Model not initialized"
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            summary = f"""
Predenergy Model Summary:
- Model Type: {'Combined (Predenergy + MoTSE)' if self.use_combined_model else 'Standard Predenergy'}
- Sequence Length: {self.seq_len}
- Horizon: {self.config.horizon}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Device: {self.device}
- Is Fitted: {self.is_fitted}
- Model Name: {self.model_name}
"""
            return summary
        except Exception as e:
            return f"Error getting model summary: {e}"

    def reset_model(self):

        try:
            self.model = None
            self.is_fitted = False
            self.optimizer = None
            self.scheduler = None
            self.criterion = None
            self.early_stopping = None
            self.data_loader = None
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            
            # 清理训练历史
            if hasattr(self, 'training_history'):
                del self.training_history
            
            print("Model state reset successfully")
        except Exception as e:
            print(f"Error resetting model: {e}")
            raise

    def set_device(self, device: str):

        try:
            self.device = torch.device(device)
            if self.model is not None:
                self.model.to(self.device)
            print(f"Model moved to device: {self.device}")
        except Exception as e:
            print(f"Error setting device: {e}")
            raise

    def get_training_history(self) -> Dict:

        try:
            if hasattr(self, 'training_history') and self.training_history is not None:
                return self.training_history
            else:
                return {'message': 'Training history not available'}
        except Exception as e:
            print(f"Error getting training history: {e}")
            return {'error': str(e)}

    def export_model(self, path: str, format: str = 'pytorch'):

        try:
            if format == 'pytorch':
                self.save_model(path)
            elif format == 'onnx':
                self._export_to_onnx(path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            print(f"Model exported successfully in {format} format to {path}")
        except Exception as e:
            print(f"Error exporting model: {e}")
            raise

    def _export_to_onnx(self, path: str):

        if self.model is None:
            raise ValueError("No model to export")
        
        try:
            # 创建示例输入
            dummy_input = torch.randn(1, self.seq_len, self.config.enc_in).to(self.device)
            
            # 导出模型
            torch.onnx.export(
                self.model,
                dummy_input,
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"Model exported to ONNX format: {path}")
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            raise

    def get_config(self) -> Dict:

        try:
            config_dict = {}
            for key, value in self.config.__dict__.items():
                if not key.startswith('_'):
                    config_dict[key] = value
            return config_dict
        except Exception as e:
            print(f"Error getting config: {e}")
            return {'error': str(e)}

    def update_config(self, **kwargs):

        updated_params = []
        invalid_params = []
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                updated_params.append(key)
            else:
                invalid_params.append(key)
        
        if updated_params:
            print(f"Updated config parameters: {updated_params}")
        
        if invalid_params:
            print(f"Warning: Invalid config parameters: {invalid_params}")
        
        # 重新验证配置
        if not self.validate_config():
            print("Warning: Configuration validation failed after update")

    def validate_config(self) -> bool:

        required_params = ['seq_len', 'horizon', 'd_model', 'n_heads']
        missing_params = []
        for param in required_params:
            if not hasattr(self.config, param):
                missing_params.append(param)
        
        if missing_params:
            print(f"Missing required parameters: {missing_params}")
            return False
        
        # 验证参数值
        if self.config.seq_len <= 0:
            print("seq_len must be positive")
            return False
        
        if self.config.horizon <= 0:
            print("horizon must be positive")
            return False
        
        if hasattr(self.config, 'd_model') and self.config.d_model <= 0:
            print("d_model must be positive")
            return False
        
        if hasattr(self.config, 'n_heads') and self.config.n_heads <= 0:
            print("n_heads must be positive")
            return False
        
        print("Configuration validation passed")
        return True

    def get_data_info(self) -> Dict:

        try:
            if self.data_loader is None:
                return {'message': 'No data loader available'}
            
            return self.data_loader.get_dataset_info()
        except Exception as e:
            print(f"Error getting data info: {e}")
            return {'error': str(e)}

    def preprocess_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        
        try:
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            # 确保数据是数值型
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("Data must be numeric")
            
            # 检查是否有NaN值
            if np.isnan(data).any():
                print("Warning: Data contains NaN values, filling with 0")
                data = np.nan_to_num(data, nan=0.0)
            
            # 标准化
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    data = self.scaler.transform(data)
                except Exception as e:
                    print(f"Warning: Error in scaling data: {e}")
                    # 如果标准化失败，使用原始数据
                    pass
            
            return data
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            raise

    def postprocess_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """后处理预测结果"""
        try:
            # 检查预测结果
            if predictions is None:
                raise ValueError("Predictions cannot be None")
            
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            # 检查是否有NaN值
            if np.isnan(predictions).any():
                print("Warning: Predictions contain NaN values, filling with 0")
                predictions = np.nan_to_num(predictions, nan=0.0)
            
            # 反标准化
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    predictions = self.scaler.inverse_transform(predictions)
                except Exception as e:
                    print(f"Warning: Error in inverse scaling predictions: {e}")
                    # 如果反标准化失败，返回原始预测
                    pass
            
            return predictions
        except Exception as e:
            print(f"Error in postprocessing predictions: {e}")
            raise

    def create_forecast_plot(self, actual: np.ndarray, predicted: np.ndarray, 
                           title: str = "Forecast vs Actual", save_path: str = None):
        """创建预测对比图"""
        try:
            import matplotlib.pyplot as plt
            
            # 检查输入数据
            if actual is None or predicted is None:
                raise ValueError("Actual and predicted data cannot be None")
            
            if len(actual) != len(predicted):
                print(f"Warning: Length mismatch - actual: {len(actual)}, predicted: {len(predicted)}")
                min_len = min(len(actual), len(predicted))
                actual = actual[:min_len]
                predicted = predicted[:min_len]
            
            plt.figure(figsize=(12, 6))
            plt.plot(actual, label='Actual', linewidth=2)
            plt.plot(predicted, label='Predicted', linewidth=2, linestyle='--')
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("Warning: matplotlib not available, cannot create plot")
        except Exception as e:
            print(f"Error creating forecast plot: {e}")
            raise

    def get_model_complexity(self) -> Dict:
        """获取模型复杂度信息"""
        try:
            if self.model is None:
                return {'message': 'Model not initialized'}
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # 计算模型大小（MB）
            param_size = 0
            buffer_size = 0
            
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': size_mb,
                'layers': len(list(self.model.modules())),
                'model_type': 'Predenergy'
            }
        except Exception as e:
            print(f"Error getting model complexity: {e}")
            return {'error': str(e)}

    def __str__(self) -> str:
        try:
            return self.get_model_summary()
        except Exception as e:
            return f"Predenergy model (error: {e})"

    def __repr__(self) -> str:
        try:
            return f"Predenergy(seq_len={self.seq_len}, horizon={getattr(self.config, 'horizon', 'N/A')}, fitted={self.is_fitted})"
        except Exception as e:
            return f"Predenergy model (error: {e})"
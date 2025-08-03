#!/usr/bin/env python
# -*- coding:utf-8 _*-
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from datasets.Predenergy_dataset import PredenergyDataset
from datasets.Predenergy_window_dataset import PredenergyWindowDataset, UniversalPredenergyWindowDataset


class PredenergyDataLoader:
    
    def __init__(
        self,
        data: Union[str, np.ndarray, pd.DataFrame],
        seq_len: int = 96,
        pred_len: int = 24,
        batch_size: int = 32,
        features: str = 'S',
        target: str = 'OT',
        timeenc: int = 0,
        freq: str = 'h',
        normalize: int = 2,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # 创建数据集
        if isinstance(data, str):
            self.dataset = PredenergyDataset(data, normalization_method='zero')
        else:
            # 如果是数组或DataFrame，先保存为临时文件
            import tempfile
            import os
            temp_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
            if isinstance(data, pd.DataFrame):
                np.save(temp_file.name, data.values)
            else:
                np.save(temp_file.name, data)
            temp_file.close()
            self.dataset = PredenergyDataset(temp_file.name, normalization_method='zero')
            # 清理临时文件
            os.unlink(temp_file.name)
        
        # 创建窗口数据集
        self.window_dataset = PredenergyWindowDataset(
            dataset=self.dataset,
            context_length=seq_len,
            prediction_length=pred_len,
            shuffle=shuffle,
            **kwargs
        )
        
        # 分割数据集
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset()
    
    def _split_dataset(self) -> Tuple[PredenergyWindowDataset, PredenergyWindowDataset, PredenergyWindowDataset]:
        """分割数据集为训练、验证和测试集"""
        total_size = len(self.window_dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.window_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
    
    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取所有数据加载器"""
        return (
            self.get_train_loader(),
            self.get_val_loader(),
            self.get_test_loader()
        )
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'total_samples': len(self.window_dataset),
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'test_samples': len(self.test_dataset),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'batch_size': self.batch_size
        }


class PredenergyUniversalDataLoader(PredenergyDataLoader):
    
    def __init__(
        self,
        data: Union[str, np.ndarray, pd.DataFrame],
        seq_len: int = 96,
        pred_len: int = 24,
        batch_size: int = 32,
        features: str = 'S',
        target: str = 'OT',
        timeenc: int = 0,
        freq: str = 'h',
        normalize: int = 2,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        shuffle: bool = True,
        num_workers: int = 0,
        use_universal: bool = True,
        max_pack_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            data=data,
            seq_len=seq_len,
            pred_len=pred_len,
            batch_size=batch_size,
            features=features,
            target=target,
            timeenc=timeenc,
            freq=freq,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
        
        if use_universal:
            # 使用通用窗口数据集
            self.window_dataset = UniversalPredenergyWindowDataset(
                dataset=self.dataset,
                context_length=seq_len,
                prediction_length=pred_len,
                shuffle=shuffle,
                **kwargs
            )
            
            # 重新分割数据集
            self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset()


def create_Predenergy_data_loader(
    data: Union[str, np.ndarray, pd.DataFrame],
    loader_type: str = 'standard',
    **kwargs
) -> PredenergyDataLoader:
    """创建Predenergy数据加载器的工厂函数"""
    if loader_type == 'standard':
        return PredenergyDataLoader(data, **kwargs)
    elif loader_type == 'universal':
        return PredenergyUniversalDataLoader(data, **kwargs)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """自定义的collate函数，用于处理批次数据"""
    
    # 收集所有键
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
    
    # 构建批次数据
    batch_data = {}
    for key in all_keys:
        if key in ['input_ids', 'labels', 'loss_masks']:
            # 对于这些键，我们需要填充到相同长度
            max_len = max(len(sample[key]) for sample in batch if key in sample)
            padded_tensors = []
            
            for sample in batch:
                if key in sample:
                    tensor = sample[key]
                    # 确保张量是一维的
                    if len(tensor.shape) == 1:
                        if len(tensor) < max_len:
                            # 填充到最大长度
                            padding_size = max_len - len(tensor)
                            if key == 'loss_masks':
                                padding = torch.zeros(padding_size, dtype=tensor.dtype)
                            else:
                                padding = torch.zeros(padding_size, dtype=tensor.dtype)
                            tensor = torch.cat([tensor, padding], dim=0)
                    else:
                        # 如果是多维张量，确保维度正确
                        if tensor.shape[0] < max_len:
                            padding_size = max_len - tensor.shape[0]
                            if key == 'loss_masks':
                                padding = torch.zeros(padding_size, *tensor.shape[1:], dtype=tensor.dtype)
                            else:
                                padding = torch.zeros(padding_size, *tensor.shape[1:], dtype=tensor.dtype)
                            tensor = torch.cat([tensor, padding], dim=0)
                    padded_tensors.append(tensor)
                else:
                    # 如果某个样本没有这个键，创建一个零张量
                    if key == 'loss_masks':
                        dummy_tensor = torch.zeros(max_len, dtype=torch.int32)
                    else:
                        dummy_tensor = torch.zeros(max_len, dtype=torch.float32)
                    padded_tensors.append(dummy_tensor)
            
            batch_data[key] = torch.stack(padded_tensors)
        else:
            # 对于其他键，直接堆叠
            tensors = [sample[key] for sample in batch if key in sample]
            if tensors:
                batch_data[key] = torch.stack(tensors)
    
    return batch_data 
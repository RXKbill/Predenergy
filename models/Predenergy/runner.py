#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import paddle
from models.modeling_Predenergy import PredenergyForPrediction
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig
from trainer.hf_trainer import PredenergyTrainingArguments, PredenergyTrainer
from datasets.Predenergy_data_loader import PredenergyDataLoader


class PredenergyRunner:
    def __init__(self, model_path: str = None, output_path: str = 'logs/Predenergy', seed: int = 9899):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed

    def load_model(self, model_path: str = None, from_scratch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path
            
        if from_scratch:
            config = PredenergyUnifiedConfig.from_pretrained(model_path)
            model = PredenergyForPrediction(config)
        else:
            model = PredenergyForPrediction.from_pretrained(model_path, **kwargs)
        return model

    def train_model(self, from_scratch: bool = False, **kwargs):
        setup_seed(self.seed)
        train_config = kwargs
        
        batch_size = train_config.get('batch_size', 32)
        precision = train_config.get('precision', 'bf16')
        
        if precision == 'bf16':
            paddle_dtype = paddle.bfloat16
        else:
            paddle_dtype = paddle.float32

        training_args = PredenergyTrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=train_config.get("num_train_epochs", 1),
            learning_rate=float(train_config.get("learning_rate", 1e-4)),
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_steps=10,
            seed=self.seed,
        )

        model_path = train_config.pop('model_path', None) or self.model_path
        if model_path is not None:
            model = self.load_model(model_path=model_path, from_scratch=from_scratch, paddle_dtype=paddle_dtype)
        else:
            raise ValueError('Model path is None')

        train_ds = self.get_train_dataset(train_config)
        
        trainer = PredenergyTrainer(model=model, args=training_args, train_dataset=train_ds)
        trainer.train()
        trainer.save_model(self.output_path)
        
        return trainer.model

    def get_train_dataset(self, train_config):
        data_path = train_config["data_path"]
        seq_len = train_config.get("seq_len", 96)
        pred_len = train_config.get("pred_len", 24)
        
        data_loader = PredenergyDataLoader(
            data=data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            batch_size=train_config.get("batch_size", 32),
        )
        
        train_loader = data_loader.get_train_loader()
        return self._convert_loader_to_dataset(train_loader)

    def _convert_loader_to_dataset(self, data_loader):
        class PredenergyHFDataset(paddle.io.Dataset):
            def __init__(self, data_loader):
                self.data = []
                for batch in data_loader:
                    if isinstance(batch, (list, tuple)):
                        x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
                    else:
                        x_enc = batch['x_enc']
                        x_mark_enc = batch['x_mark_enc']
                        x_dec = batch['x_dec']
                        x_mark_dec = batch['x_mark_dec']
                        y = batch['y']
                    
                    self.data.append({
                        'x_enc': x_enc,
                        'x_mark_enc': x_mark_enc,
                        'x_dec': x_dec,
                        'x_mark_dec': x_mark_dec,
                        'labels': y
                    })
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return PredenergyHFDataset(data_loader)


def setup_seed(seed: int = 9899):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import paddle
        paddle.seed(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number) 
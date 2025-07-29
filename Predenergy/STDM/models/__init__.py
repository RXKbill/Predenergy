# -*- coding: utf-8 -*-

__all__ = [
    "ModelFactory",
    "ModelBase",
    "get_models",
]

ADAPTER = {
    "darts_deep_model_adapter": "ts_benchmark.baselines.darts.darts_deep_model_adapter",
    "darts_statistical_model_adapter": "ts_benchmark.baselines.darts.darts_statistical_model_adapter",
    "darts_regression_model_adapter": "ts_benchmark.baselines.darts.darts_regression_model_adapter",
    "transformer_adapter": "ts_benchmark.baselines.time_series_library.adapters_for_transformers.transformer_adapter",
}

from model_base import ModelBase
from model_loader import ModelFactory, get_models

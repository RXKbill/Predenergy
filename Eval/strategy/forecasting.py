# -*- coding: utf-8 -*-
import abc
import traceback
from typing import Any, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.data_pool import DataPool
from Eval.strategy.constants import FieldNames
from Eval.strategy.strategy import Strategy
from models.models import ModelFactory
from utils.data_processing import split_before
from utils.random_utils import fix_random_seed, fix_all_random_seed


class ForecastingStrategy(Strategy, metaclass=abc.ABCMeta):


    REQUIRED_CONFIGS = [
        "seed",
        "deterministic"
    ]

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:

        deterministic_mode = self._get_scalar_config_value("deterministic", series_name)
        seed = self._get_scalar_config_value("seed", series_name)

        if deterministic_mode == "full":
            fix_all_random_seed(seed)
        elif deterministic_mode == "efficient":
            fix_random_seed(seed)

        data_pool = DataPool().get_pool()
        data = data_pool.get_series(series_name)
        meta_info = data_pool.get_series_meta_info(series_name)

        try:
            single_series_results = self._execute(
                data, meta_info, model_factory, series_name
            )
        except Exception as e:
            log = f"{traceback.format_exc()}\n{e}"
            single_series_results = self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )

        return single_series_results

    @abc.abstractmethod
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> Any:
        pass

    def _get_eval_scaler(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> Any:
        train_data, _ = split_before(
            train_valid_data,
            int(len(train_valid_data) * train_ratio_in_tv),
        )
        scaler = StandardScaler().fit(train_data.values)
        return scaler

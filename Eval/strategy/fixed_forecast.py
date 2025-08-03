# -*- coding: utf-8 -*-
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from Eval.metrics import regression_metrics
from Eval.strategy.constants import FieldNames
from Eval.strategy.forecasting import ForecastingStrategy
from models.models import ModelFactory
from utils.data_processing import split_before


class FixedForecast(ForecastingStrategy):

    REQUIRED_CONFIGS = [
        "horizon",
        "train_ratio_in_tv",
        "save_true_pred",
    ]

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        model = model_factory()

        horizon = self._get_scalar_config_value("horizon", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )

        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = data_len - horizon
        if train_length <= 0:
            raise ValueError("The prediction step exceeds the data length")

        train_valid_data, test_data = split_before(series, train_length)
        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
        end_fit_time = time.time()
        predicted = model.forecast(horizon, train_valid_data)
        end_inference_time = time.time()

        single_series_results, log_info = self.evaluator.evaluate_with_log(
            test_data.to_numpy(),
            predicted,
            # TODO: add configs to control scaling behavior
            self._get_eval_scaler(train_valid_data, train_ratio_in_tv),
            train_valid_data.values,
        )
        inference_data = pd.DataFrame(
            predicted, columns=test_data.columns, index=test_data.index
        )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = self._encode_data(test_data) if save_true_pred else np.nan
        inference_data_encoded = self._encode_data(inference_data) if save_true_pred else np.nan

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            end_inference_time - end_fit_time,
            actual_data_encoded,
            inference_data_encoded,
            log_info,
        ]

        return single_series_results

    @staticmethod
    def accepted_metrics():
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]

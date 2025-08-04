# -*- coding: utf-8 -*-
from Eval.strategy.forecast import FixedForecast
from Eval.strategy.rolling_forecast import RollingForecast

STRATEGY = {
    "forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
}
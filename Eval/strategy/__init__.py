# -*- coding: utf-8 -*-
from Eval.strategy.fixed_forecast import FixedForecast
from Eval.strategy.rolling_forecast import RollingForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
}
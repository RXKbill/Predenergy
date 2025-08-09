from typing import List
import paddle
import warnings

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# PaddleTS集成
try:
    from paddlets.datasets.tsdataset import TSDataset
    from paddlets.transform.time_feature import TimeFeatureGenerator
    PADDLETS_AVAILABLE = True
except ImportError:
    PADDLETS_AVAILABLE = False
    warnings.warn("PaddleTS not available. Some advanced time features will be disabled.")


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def paddlets_time_features(dates, freq='h', use_cyclical=True):
    """
    使用PaddleTS生成高级时间特征
    
    Args:
        dates: 日期索引
        freq: 频率字符串
        use_cyclical: 是否使用循环编码
    
    Returns:
        np.ndarray: 时间特征矩阵
    """
    if not PADDLETS_AVAILABLE:
        # 降级到基础时间特征
        return time_features(dates, freq)
    
    try:
        # 创建简单的时序数据
        df = pd.DataFrame(index=dates, data={'value': np.ones(len(dates))})
        ts_data = TSDataset.load_from_dataframe(df, target_cols=['value'])
        
        # 配置时间特征生成器
        time_features_config = {
            'hour_of_day': True,
            'day_of_week': True,
            'day_of_month': True,
            'day_of_year': True,
            'month_of_year': True,
            'week_of_year': True,
        }
        
        if use_cyclical:
            # 添加循环编码
            time_features_config.update({
                'hour_of_day_cyclical': True,
                'day_of_week_cyclical': True,
                'month_of_year_cyclical': True,
            })
        
        feature_generator = TimeFeatureGenerator(**time_features_config)
        ts_data_with_features = feature_generator.fit_transform(ts_data)
        
        # 提取特征（排除原始的value列）
        feature_df = ts_data_with_features.to_dataframe()
        feature_columns = [col for col in feature_df.columns if col != 'value']
        features = feature_df[feature_columns].values.T
        
        return features
        
    except Exception as e:
        warnings.warn(f"PaddleTS time feature extraction failed: {e}. Using basic features.")
        return time_features(dates, freq)


def enhanced_time_features(dates, freq='h', method='paddlets'):
    """
    增强的时间特征提取函数
    
    Args:
        dates: 日期索引
        freq: 频率字符串
        method: 特征提取方法 ('basic', 'paddlets')
    
    Returns:
        np.ndarray: 时间特征矩阵
    """
    if method == 'paddlets' and PADDLETS_AVAILABLE:
        return paddlets_time_features(dates, freq)
    else:
        return time_features(dates, freq)

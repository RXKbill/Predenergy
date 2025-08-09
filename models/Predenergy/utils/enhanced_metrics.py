"""
Enhanced Evaluation Metrics for Time Series Forecasting
This module provides comprehensive evaluation metrics beyond the basic ones.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.metrics import r2_score
import warnings


class EnhancedForeccastMetrics:
    """
    Comprehensive evaluation metrics for time series forecasting.
    
    Includes statistical, probabilistic, and domain-specific metrics.
    """
    
    @staticmethod
    def basic_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate basic regression metrics."""
        
        # Handle potential division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            metrics = {
                'mse': np.average((y_true - y_pred) ** 2, weights=sample_weight),
                'rmse': np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight)),
                'mae': np.average(np.abs(y_true - y_pred), weights=sample_weight),
                'mape': np.average(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None)) * 100, weights=sample_weight),
                'r2': r2_score(y_true, y_pred, sample_weight=sample_weight),
            }
            
            # Handle inf/nan values
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    metrics[key] = float('inf') if 'mape' in key else 0.0
                    
        return metrics
    
    @staticmethod
    def advanced_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate advanced forecasting metrics."""
        
        residuals = y_true - y_pred
        
        metrics = {
            # Symmetric metrics
            'smape': np.average(
                2 * np.abs(residuals) / (np.abs(y_true) + np.abs(y_pred) + 1e-8) * 100,
                weights=sample_weight
            ),
            
            # Weighted metrics
            'wape': np.sum(np.abs(residuals)) / (np.sum(np.abs(y_true)) + 1e-8) * 100,
            
            # Normalized metrics
            'nrmse': np.sqrt(np.average(residuals ** 2, weights=sample_weight)) / (np.std(y_true) + 1e-8),
            'nmae': np.average(np.abs(residuals), weights=sample_weight) / (np.mean(np.abs(y_true)) + 1e-8),
            
            # Directional accuracy
            'da': EnhancedForeccastMetrics._directional_accuracy(y_true, y_pred),
            
            # Theil's U statistic
            'theil_u': EnhancedForeccastMetrics._theil_u(y_true, y_pred),
            
            # Mean Absolute Scaled Error (requires seasonal data)
            'mase': EnhancedForeccastMetrics._mase(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def probabilistic_metrics(
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None,
        quantile_predictions: Optional[Dict[float, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate probabilistic forecasting metrics.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted mean values
            y_pred_std: Predicted standard deviations
            quantile_predictions: Dictionary of quantile predictions {quantile: predictions}
        """
        metrics = {}
        
        if y_pred_std is not None:
            # Continuous Ranked Probability Score
            metrics['crps'] = EnhancedForeccastMetrics._crps_gaussian(y_true, y_pred_mean, y_pred_std)
            
            # Log-likelihood (assuming Gaussian)
            metrics['log_likelihood'] = EnhancedForeccastMetrics._gaussian_log_likelihood(y_true, y_pred_mean, y_pred_std)
            
            # Prediction Interval Coverage Probability
            for confidence in [0.8, 0.9, 0.95]:
                lower, upper = EnhancedForeccastMetrics._prediction_interval(y_pred_mean, y_pred_std, confidence)
                coverage = np.mean((y_true >= lower) & (y_true <= upper))
                metrics[f'picp_{int(confidence*100)}'] = coverage
        
        if quantile_predictions is not None:
            # Quantile Score
            for q, y_pred_q in quantile_predictions.items():
                metrics[f'qs_{int(q*100)}'] = EnhancedForeccastMetrics._quantile_score(y_true, y_pred_q, q)
            
            # Interval Score (if 50% quantiles available)
            if 0.25 in quantile_predictions and 0.75 in quantile_predictions:
                metrics['interval_score_50'] = EnhancedForeccastMetrics._interval_score(
                    y_true, quantile_predictions[0.25], quantile_predictions[0.75], 0.5
                )
        
        return metrics
    
    @staticmethod
    def distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate distribution comparison metrics."""
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(y_true, y_pred)
        
        # Wasserstein distance (Earth Mover's Distance)
        wasserstein_dist = stats.wasserstein_distance(y_true, y_pred)
        
        # Energy distance
        energy_dist = EnhancedForeccastMetrics._energy_distance(y_true, y_pred)
        
        # Distribution moments comparison
        moments_diff = {
            'mean_diff': np.abs(np.mean(y_true) - np.mean(y_pred)),
            'std_diff': np.abs(np.std(y_true) - np.std(y_pred)),
            'skew_diff': np.abs(stats.skew(y_true) - stats.skew(y_pred)),
            'kurtosis_diff': np.abs(stats.kurtosis(y_true) - stats.kurtosis(y_pred)),
        }
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'wasserstein_distance': wasserstein_dist,
            'energy_distance': energy_dist,
            **moments_diff
        }
    
    @staticmethod
    def temporal_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        freq: str = 'H'
    ) -> Dict[str, float]:
        """Calculate temporal pattern evaluation metrics."""
        
        # Autocorrelation similarity
        acf_true = EnhancedForeccastMetrics._autocorrelation(y_true)
        acf_pred = EnhancedForeccastMetrics._autocorrelation(y_pred)
        acf_similarity = np.corrcoef(acf_true, acf_pred)[0, 1]
        
        # Spectral similarity (frequency domain)
        spectral_similarity = EnhancedForeccastMetrics._spectral_similarity(y_true, y_pred)
        
        # Trend similarity
        trend_similarity = EnhancedForeccastMetrics._trend_similarity(y_true, y_pred)
        
        # Seasonality similarity (if applicable)
        seasonality_similarity = EnhancedForeccastMetrics._seasonality_similarity(y_true, y_pred, freq)
        
        return {
            'acf_similarity': acf_similarity if not np.isnan(acf_similarity) else 0.0,
            'spectral_similarity': spectral_similarity,
            'trend_similarity': trend_similarity,
            'seasonality_similarity': seasonality_similarity,
        }
    
    @staticmethod
    def comprehensive_evaluation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None,
        quantile_predictions: Optional[Dict[float, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        freq: str = 'H'
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Comprehensive evaluation combining all metric types.
        
        Returns:
            Dictionary with categorized metrics
        """
        
        evaluation = {
            'basic': EnhancedForeccastMetrics.basic_metrics(y_true, y_pred, sample_weight),
            'advanced': EnhancedForeccastMetrics.advanced_metrics(y_true, y_pred, sample_weight),
            'distribution': EnhancedForeccastMetrics.distribution_metrics(y_true, y_pred),
            'temporal': EnhancedForeccastMetrics.temporal_metrics(y_true, y_pred, freq),
        }
        
        if y_pred_std is not None or quantile_predictions is not None:
            evaluation['probabilistic'] = EnhancedForeccastMetrics.probabilistic_metrics(
                y_true, y_pred, y_pred_std, quantile_predictions
            )
        
        return evaluation
    
    # Helper methods
    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy."""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction)
    
    @staticmethod
    def _theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        
        return numerator / (denominator + 1e-8)
    
    @staticmethod
    def _mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 1) -> float:
        """Calculate Mean Absolute Scaled Error."""
        if len(y_true) <= seasonality:
            return float('inf')
        
        naive_forecast = y_true[:-seasonality]
        naive_mae = np.mean(np.abs(y_true[seasonality:] - naive_forecast))
        
        forecast_mae = np.mean(np.abs(y_true - y_pred))
        
        return forecast_mae / (naive_mae + 1e-8)
    
    @staticmethod
    def _crps_gaussian(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> float:
        """Calculate CRPS for Gaussian predictions."""
        standardized = (y_true - y_pred_mean) / (y_pred_std + 1e-8)
        
        crps = y_pred_std * (
            standardized * (2 * stats.norm.cdf(standardized) - 1) +
            2 * stats.norm.pdf(standardized) - 1/np.sqrt(np.pi)
        )
        
        return np.mean(crps)
    
    @staticmethod
    def _gaussian_log_likelihood(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> float:
        """Calculate Gaussian log-likelihood."""
        return np.mean(stats.norm.logpdf(y_true, y_pred_mean, y_pred_std + 1e-8))
    
    @staticmethod
    def _prediction_interval(y_pred_mean: np.ndarray, y_pred_std: np.ndarray, confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lower = y_pred_mean - z_score * y_pred_std
        upper = y_pred_mean + z_score * y_pred_std
        return lower, upper
    
    @staticmethod
    def _quantile_score(y_true: np.ndarray, y_pred_q: np.ndarray, quantile: float) -> float:
        """Calculate quantile score."""
        error = y_true - y_pred_q
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    @staticmethod
    def _interval_score(y_true: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray, alpha: float) -> float:
        """Calculate interval score."""
        width = y_pred_upper - y_pred_lower
        lower_violation = 2 * alpha * np.maximum(0, y_pred_lower - y_true)
        upper_violation = 2 * alpha * np.maximum(0, y_true - y_pred_upper)
        
        return np.mean(width + lower_violation + upper_violation)
    
    @staticmethod
    def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate energy distance between two samples."""
        n, m = len(x), len(y)
        
        # Calculate pairwise distances
        xx = np.mean([np.abs(x[i] - x[j]) for i in range(n) for j in range(n)])
        yy = np.mean([np.abs(y[i] - y[j]) for i in range(m) for j in range(m)])
        xy = np.mean([np.abs(x[i] - y[j]) for i in range(n) for j in range(m)])
        
        return 2 * xy - xx - yy
    
    @staticmethod
    def _autocorrelation(x: np.ndarray, max_lags: int = 10) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(x)
        x = x - np.mean(x)
        
        autocorr = []
        for lag in range(min(max_lags, n-1)):
            if n - lag > 1:
                corr = np.corrcoef(x[:-lag] if lag > 0 else x, x[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0.0)
            else:
                autocorr.append(0.0)
        
        return np.array(autocorr)
    
    @staticmethod
    def _spectral_similarity(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate spectral similarity using power spectral densities."""
        try:
            from scipy.signal import welch
            
            f_x, psd_x = welch(x)
            f_y, psd_y = welch(y)
            
            # Normalize PSDs
            psd_x = psd_x / np.sum(psd_x)
            psd_y = psd_y / np.sum(psd_y)
            
            # Calculate correlation between normalized PSDs
            correlation = np.corrcoef(psd_x, psd_y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    @staticmethod
    def _trend_similarity(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate trend similarity using linear regression slopes."""
        try:
            t = np.arange(len(x))
            
            # Calculate slopes
            slope_x = np.polyfit(t, x, 1)[0]
            slope_y = np.polyfit(t, y, 1)[0]
            
            # Normalize by standard deviations
            slope_x_norm = slope_x / (np.std(x) + 1e-8)
            slope_y_norm = slope_y / (np.std(y) + 1e-8)
            
            # Calculate similarity (1 - normalized difference)
            return 1 - min(1, abs(slope_x_norm - slope_y_norm))
            
        except Exception:
            return 0.0
    
    @staticmethod
    def _seasonality_similarity(x: np.ndarray, y: np.ndarray, freq: str = 'H') -> float:
        """Calculate seasonality similarity."""
        try:
            # Determine seasonal period based on frequency
            seasonal_periods = {'H': 24, 'D': 7, 'W': 52, 'M': 12}
            period = seasonal_periods.get(freq, 24)
            
            if len(x) < 2 * period:
                return 0.0
            
            # Extract seasonal components (simple seasonal decomposition)
            def extract_seasonal(data, period):
                seasonal = []
                for i in range(period):
                    seasonal.append(np.mean(data[i::period]))
                return np.array(seasonal)
            
            seasonal_x = extract_seasonal(x, period)
            seasonal_y = extract_seasonal(y, period)
            
            # Calculate correlation between seasonal components
            correlation = np.corrcoef(seasonal_x, seasonal_y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0


def print_metrics_summary(metrics: Dict[str, Union[float, Dict[str, float]]], title: str = "Metrics Summary"):
    """Pretty print metrics summary."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for category, category_metrics in metrics.items():
        print(f"\n{category.upper()} METRICS:")
        print("-" * 40)
        
        if isinstance(category_metrics, dict):
            for metric, value in category_metrics.items():
                print(f"{metric:25}: {value:10.4f}")
        else:
            print(f"{category:25}: {category_metrics:10.4f}")
    
    print(f"\n{'='*60}\n")
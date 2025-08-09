"""
Predenergy Visualization Tools
This module provides comprehensive visualization capabilities for time series forecasting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple, Union
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PredenergyVisualizer:
    """
    Comprehensive visualization toolkit for Predenergy forecasting results.
    Supports both static (matplotlib/seaborn) and interactive (plotly) visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'actual': '#2E86C1',
            'predicted': '#E74C3C',
            'confidence': '#F39C12',
            'background': '#F8F9FA'
        }
    
    def plot_forecast_comparison(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[np.ndarray] = None,
        confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        title: str = "Forecast Comparison",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> None:
        """
        Plot actual vs predicted values with optional confidence intervals.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            dates: Date index for x-axis
            confidence_interval: Tuple of (lower_bound, upper_bound)
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive plotly plot
        """
        if interactive:
            self._plot_forecast_interactive(
                actual, predicted, dates, confidence_interval, title, save_path
            )
        else:
            self._plot_forecast_static(
                actual, predicted, dates, confidence_interval, title, save_path
            )
    
    def _plot_forecast_static(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[np.ndarray],
        confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]],
        title: str,
        save_path: Optional[str]
    ) -> None:
        """Create static matplotlib plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = dates if dates is not None else np.arange(len(actual))
        
        # Plot actual and predicted
        ax.plot(x, actual, label='Actual', color=self.colors['actual'], linewidth=2)
        ax.plot(x, predicted, label='Predicted', color=self.colors['predicted'], linewidth=2)
        
        # Plot confidence interval
        if confidence_interval is not None:
            lower, upper = confidence_interval
            ax.fill_between(
                x, lower, upper,
                alpha=0.3, color=self.colors['confidence'],
                label='Confidence Interval'
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_forecast_interactive(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[np.ndarray],
        confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]],
        title: str,
        save_path: Optional[str]
    ) -> None:
        """Create interactive plotly plot."""
        fig = go.Figure()
        
        x = dates if dates is not None else np.arange(len(actual))
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=x, y=actual,
            mode='lines',
            name='Actual',
            line=dict(color=self.colors['actual'], width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=x, y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color=self.colors['predicted'], width=2)
        ))
        
        # Add confidence interval
        if confidence_interval is not None:
            lower, upper = confidence_interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor=f'rgba(243, 156, 18, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot residual analysis including residuals over time and distribution.
        """
        residuals = actual - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        x = dates if dates is not None else np.arange(len(actual))
        
        # Residuals over time
        axes[0, 0].plot(x, residuals, color=self.colors['actual'], alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=self.colors['predicted'])
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Actual vs Predicted scatter
        axes[1, 1].scatter(actual, predicted, alpha=0.6, color=self.colors['confidence'])
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        input_labels: Optional[List[str]] = None,
        output_labels: Optional[List[str]] = None,
        title: str = "Attention Weights Heatmap",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention_weights: Attention matrix [seq_len, seq_len] or [heads, seq_len, seq_len]
            input_labels: Labels for input sequence
            output_labels: Labels for output sequence
            title: Plot title
            save_path: Path to save the plot
        """
        if attention_weights.ndim == 3:
            # Average over heads
            attention_weights = attention_weights.mean(axis=0)
        
        plt.figure(figsize=self.figsize)
        
        sns.heatmap(
            attention_weights,
            xticklabels=input_labels,
            yticklabels=output_labels,
            cmap='Blues',
            annot=False,
            cbar=True
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Input Sequence')
        plt.ylabel('Output Sequence')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history including loss curves and metrics.
        
        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 6))
        
        if len(history) == 1:
            axes = [axes]
        
        for idx, (metric, values) in enumerate(history.items()):
            axes[idx].plot(values, color=self.colors['actual'], linewidth=2)
            axes[idx].set_title(f'{metric.title()} Over Epochs')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.title())
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_forecast_dashboard(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        metrics: Dict[str, float],
        attention_weights: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create subplots
        if attention_weights is not None:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Forecast Comparison',
                    'Residuals Over Time',
                    'Attention Weights',
                    'Performance Metrics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
        else:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Forecast Comparison',
                    'Residuals Over Time',
                    'Actual vs Predicted',
                    'Performance Metrics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
        
        x = dates if dates is not None else np.arange(len(actual))
        residuals = actual - predicted
        
        # Forecast comparison
        fig.add_trace(
            go.Scatter(x=x, y=actual, name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=predicted, name='Predicted', line=dict(color='red')),
            row=1, col=1
        )
        
        # Residuals
        fig.add_trace(
            go.Scatter(x=x, y=residuals, name='Residuals', line=dict(color='green')),
            row=1, col=2
        )
        
        # Actual vs Predicted or Attention Weights
        if attention_weights is not None:
            fig.add_trace(
                go.Heatmap(z=attention_weights.mean(axis=0) if attention_weights.ndim == 3 else attention_weights),
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=actual, y=predicted, mode='markers',
                    name='Actual vs Predicted', marker=dict(color='orange')
                ),
                row=2, col=1
            )
        
        # Metrics table
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[list(metrics.keys()), [f"{v:.4f}" for v in metrics.values()]])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Predenergy Forecast Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()


def quick_forecast_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Quick Forecast Plot"
) -> None:
    """Quick utility function for simple forecast plotting."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', linewidth=2)
    plt.plot(predicted, label='Predicted', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importance_scores: Dict[str, float],
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> None:
    """Plot feature importance scores."""
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, scores)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
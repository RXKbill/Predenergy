"""
Unit tests for Predenergy models.
"""

import unittest
import numpy as np
import paddle
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.Predenergy.models.unified_config import PredenergyUnifiedConfig
from models.Predenergy.models.predenergy_model import PredenergyModel, PredenergyForPrediction
from models.Predenergy.utils.enhanced_metrics import EnhancedForeccastMetrics
from models.Predenergy.utils.visualization import PredenergyVisualizer


class TestPredenergyUnifiedConfig(unittest.TestCase):
    """Test the unified configuration system."""
    
    def test_config_creation(self):
        """Test basic configuration creation."""
        config = PredenergyUnifiedConfig()
        
        # Test default values
        self.assertEqual(config.seq_len, 96)
        self.assertEqual(config.horizon, 24)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_heads, 8)
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = PredenergyUnifiedConfig(seq_len=96, horizon=24, d_model=512, n_heads=8)
        self.assertTrue(True)  # Should not raise exception
        
        # Test invalid configuration
        with self.assertRaises(ValueError):
            PredenergyUnifiedConfig(seq_len=-1)
        
        with self.assertRaises(ValueError):
            PredenergyUnifiedConfig(d_model=100, n_heads=8)  # Not divisible
    
    def test_config_methods(self):
        """Test configuration utility methods."""
        config = PredenergyUnifiedConfig()
        
        # Test getting sub-configurations
        motse_config = config.get_motse_config()
        self.assertIn('hidden_size', motse_config)
        self.assertIn('num_experts', motse_config)
        
        stdm_config = config.get_stdm_config()
        self.assertIn('seq_len', stdm_config)
        self.assertIn('d_model', stdm_config)
        
        decoder_config = config.get_decoder_config()
        self.assertIn('hidden_size', decoder_config)
        self.assertIn('num_layers', decoder_config)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = PredenergyUnifiedConfig(seq_len=128, horizon=48)
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['seq_len'], 128)
        self.assertEqual(config_dict['horizon'], 48)
        
        # Test from_dict
        new_config = PredenergyUnifiedConfig.from_dict(config_dict)
        self.assertEqual(new_config.seq_len, 128)
        self.assertEqual(new_config.horizon, 48)


class TestPredenergyModel(unittest.TestCase):
    """Test the main Predenergy model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PredenergyUnifiedConfig(
            seq_len=24,
            horizon=6,
            d_model=64,
            n_heads=4,
            e_layers=1,
            motse_hidden_size=64,
            motse_num_layers=1,
            motse_num_heads=4,
            decoder_hidden_size=64,
            decoder_num_layers=1,
            use_paddlenlp_decoder=False  # Disable for testing
        )
        
        self.batch_size = 2
        self.seq_len = self.config.seq_len
        self.input_size = self.config.input_size
        
        # Create sample input data
        self.sample_input = paddle.randn([self.batch_size, self.seq_len, self.input_size])
    
    def test_model_creation(self):
        """Test model instantiation."""
        model = PredenergyModel(self.config)
        self.assertIsInstance(model, PredenergyModel)
        
        # Test model components
        self.assertIsNotNone(model.connection)
        self.assertIsNotNone(model.motse_model)
        self.assertIsNotNone(model.cluster_model)
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = PredenergyModel(self.config)
        
        # Test forward pass
        outputs = model(self.sample_input)
        
        # Check output structure
        self.assertIn('predictions', outputs)
        self.assertIn('L_importance', outputs)
        
        # Check output shapes
        predictions = outputs['predictions']
        expected_shape = [self.batch_size, self.config.horizon, self.config.c_out]
        self.assertEqual(list(predictions.shape), expected_shape)
    
    def test_model_with_decoder(self):
        """Test model with PaddleNLP decoder (if available)."""
        try:
            config_with_decoder = PredenergyUnifiedConfig(
                seq_len=24,
                horizon=6,
                d_model=64,
                n_heads=4,
                e_layers=1,
                motse_hidden_size=64,
                motse_num_layers=1,
                motse_num_heads=4,
                decoder_hidden_size=64,
                decoder_num_layers=1,
                use_paddlenlp_decoder=True
            )
            
            model = PredenergyModel(config_with_decoder)
            outputs = model(self.sample_input)
            
            self.assertIn('predictions', outputs)
            self.assertTrue(outputs['decoder_used'])
            
        except ImportError:
            self.skipTest("PaddleNLP not available for decoder testing")


class TestPredenergyForPrediction(unittest.TestCase):
    """Test the prediction wrapper model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PredenergyUnifiedConfig(
            seq_len=24,
            horizon=6,
            d_model=64,
            n_heads=4,
            e_layers=1,
            motse_hidden_size=64,
            motse_num_layers=1,
            motse_num_heads=4,
            decoder_hidden_size=64,
            decoder_num_layers=1,
            use_paddlenlp_decoder=False
        )
        
        self.batch_size = 2
        self.seq_len = self.config.seq_len
        self.input_size = self.config.input_size
        
        self.sample_input = paddle.randn([self.batch_size, self.seq_len, self.input_size])
        self.sample_labels = paddle.randn([self.batch_size, self.config.horizon, self.config.c_out])
    
    def test_prediction_model_creation(self):
        """Test prediction model instantiation."""
        model = PredenergyForPrediction(self.config)
        self.assertIsInstance(model, PredenergyForPrediction)
    
    def test_prediction_forward_without_labels(self):
        """Test forward pass without labels (inference mode)."""
        model = PredenergyForPrediction(self.config)
        
        outputs = model(self.sample_input)
        
        self.assertIn('predictions', outputs)
        self.assertIsNone(outputs['loss'])
        
        predictions = outputs['predictions']
        expected_shape = [self.batch_size, self.config.horizon, self.config.c_out]
        self.assertEqual(list(predictions.shape), expected_shape)
    
    def test_prediction_forward_with_labels(self):
        """Test forward pass with labels (training mode)."""
        model = PredenergyForPrediction(self.config)
        
        outputs = model(self.sample_input, labels=self.sample_labels)
        
        self.assertIn('predictions', outputs)
        self.assertIn('loss', outputs)
        self.assertIsNotNone(outputs['loss'])
        
        # Check that loss is a scalar
        self.assertEqual(len(outputs['loss'].shape), 0)
    
    def test_prediction_methods(self):
        """Test prediction utility methods."""
        model = PredenergyForPrediction(self.config)
        
        # Test predict method
        predictions = model.predict(self.sample_input)
        expected_shape = [self.batch_size, self.config.horizon, self.config.c_out]
        self.assertEqual(list(predictions.shape), expected_shape)


class TestEnhancedMetrics(unittest.TestCase):
    """Test the enhanced metrics module."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.random.randn(100)
        self.y_pred = self.y_true + 0.1 * np.random.randn(100)  # Add some noise
        self.y_pred_std = np.ones_like(self.y_pred) * 0.1
    
    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        metrics = EnhancedForeccastMetrics.basic_metrics(self.y_true, self.y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertFalse(np.isnan(metrics[metric]))
    
    def test_advanced_metrics(self):
        """Test advanced metrics calculation."""
        metrics = EnhancedForeccastMetrics.advanced_metrics(self.y_true, self.y_pred)
        
        expected_metrics = ['smape', 'wape', 'nrmse', 'nmae', 'da', 'theil_u', 'mase']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_probabilistic_metrics(self):
        """Test probabilistic metrics calculation."""
        metrics = EnhancedForeccastMetrics.probabilistic_metrics(
            self.y_true, self.y_pred, self.y_pred_std
        )
        
        expected_metrics = ['crps', 'log_likelihood']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        evaluation = EnhancedForeccastMetrics.comprehensive_evaluation(
            self.y_true, self.y_pred, self.y_pred_std
        )
        
        # Check that all metric categories are present
        expected_categories = ['basic', 'advanced', 'distribution', 'temporal', 'probabilistic']
        for category in expected_categories:
            self.assertIn(category, evaluation)
            self.assertIsInstance(evaluation[category], dict)


class TestVisualization(unittest.TestCase):
    """Test the visualization module."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.actual = np.random.randn(100)
        self.predicted = self.actual + 0.1 * np.random.randn(100)
        self.dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Create temporary directory for saving plots
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_visualizer_creation(self):
        """Test visualizer instantiation."""
        visualizer = PredenergyVisualizer()
        self.assertIsInstance(visualizer, PredenergyVisualizer)
    
    def test_forecast_comparison_plot(self):
        """Test forecast comparison plotting."""
        visualizer = PredenergyVisualizer()
        
        # Test without saving (just check it doesn't crash)
        try:
            # Use matplotlib backend that doesn't require display
            import matplotlib
            matplotlib.use('Agg')
            
            visualizer.plot_forecast_comparison(
                self.actual, self.predicted, dates=self.dates[:100],
                title="Test Forecast"
            )
            
        except Exception as e:
            self.fail(f"Forecast comparison plot failed: {e}")
    
    def test_residuals_plot(self):
        """Test residuals plotting."""
        visualizer = PredenergyVisualizer()
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            visualizer.plot_residuals(self.actual, self.predicted)
            
        except Exception as e:
            self.fail(f"Residuals plot failed: {e}")


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete model pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = PredenergyUnifiedConfig(
            seq_len=24,
            horizon=6,
            d_model=64,
            n_heads=4,
            e_layers=1,
            motse_hidden_size=64,
            motse_num_layers=1,
            motse_num_heads=4,
            decoder_hidden_size=64,
            decoder_num_layers=1,
            use_paddlenlp_decoder=False,
            batch_size=4
        )
        
        # Create sample dataset
        self.n_samples = 50
        self.train_data = np.random.randn(self.n_samples, self.config.seq_len, self.config.input_size)
        self.test_data = np.random.randn(10, self.config.seq_len, self.config.input_size)
        self.labels = np.random.randn(10, self.config.horizon, self.config.c_out)
    
    def test_end_to_end_training(self):
        """Test end-to-end training simulation."""
        model = PredenergyForPrediction(self.config)
        
        # Convert to tensors
        train_input = paddle.to_tensor(self.train_data[:10], dtype='float32')
        train_labels = paddle.to_tensor(np.random.randn(10, self.config.horizon, self.config.c_out), dtype='float32')
        
        # Simulate training step
        model.train()
        outputs = model(train_input, labels=train_labels)
        
        # Check that loss is computed
        self.assertIsNotNone(outputs['loss'])
        self.assertTrue(outputs['loss'].item() >= 0)
    
    def test_end_to_end_inference(self):
        """Test end-to-end inference."""
        model = PredenergyForPrediction(self.config)
        
        # Convert to tensors
        test_input = paddle.to_tensor(self.test_data, dtype='float32')
        
        # Simulate inference
        model.eval()
        with paddle.no_grad():
            outputs = model(test_input)
        
        # Check outputs
        self.assertIn('predictions', outputs)
        predictions = outputs['predictions']
        
        # Check shapes
        expected_shape = [len(self.test_data), self.config.horizon, self.config.c_out]
        self.assertEqual(list(predictions.shape), expected_shape)
    
    def test_metrics_evaluation(self):
        """Test metrics evaluation on predictions."""
        model = PredenergyForPrediction(self.config)
        
        # Generate predictions
        test_input = paddle.to_tensor(self.test_data, dtype='float32')
        
        model.eval()
        with paddle.no_grad():
            outputs = model(test_input)
        
        predictions = outputs['predictions'].numpy()
        
        # Evaluate metrics
        y_true = self.labels.flatten()
        y_pred = predictions.flatten()
        
        metrics = EnhancedForeccastMetrics.basic_metrics(y_true, y_pred)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestPredenergyUnifiedConfig,
        TestPredenergyModel,
        TestPredenergyForPrediction,
        TestEnhancedMetrics,
        TestVisualization,
        TestModelIntegration,
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import pandas as pd  # Import pandas for the tests
    
    print("Running Predenergy Model Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    exit(0 if success else 1)
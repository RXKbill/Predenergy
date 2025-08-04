# -*- coding: utf-8 -*-
import os

# Get the root path where the code file is located
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))

# Dataset and Data Paths
DATASET_PATH = os.path.join(ROOT_PATH, "dataset")
DATA_PATH = os.path.join(ROOT_PATH, "data")

# Model Paths
MODELS_PATH = os.path.join(ROOT_PATH, "models")
PREDENERGY_PATH = os.path.join(MODELS_PATH, "Predenergy")
PREDENERGY_DATASETS_PATH = os.path.join(PREDENERGY_PATH, "datasets")
PREDENERGY_LAYERS_PATH = os.path.join(PREDENERGY_PATH, "layers")
PREDENERGY_MODELS_PATH = os.path.join(PREDENERGY_PATH, "models")
PREDENERGY_TRAINER_PATH = os.path.join(PREDENERGY_PATH, "trainer")
PREDENERGY_UTILS_PATH = os.path.join(PREDENERGY_PATH, "utils")

# Configuration Paths
CONFIG_PATH = os.path.join(ROOT_PATH, "configs")

# Evaluation Paths
EVAL_PATH = os.path.join(ROOT_PATH, "Eval")
EVAL_METRICS_PATH = os.path.join(EVAL_PATH, "metrics")
EVAL_STRATEGY_PATH = os.path.join(EVAL_PATH, "strategy")

# Utility Paths
UTILS_PATH = os.path.join(ROOT_PATH, "utils")
UTILS_PARALLEL_PATH = os.path.join(UTILS_PATH, "parallel")

# LLaMA-Factory Paths
LLAMA_FACTORY_PATH = os.path.join(ROOT_PATH, "LLaMA-Factory")
LLAMA_FACTORY_EXAMPLES_PATH = os.path.join(LLAMA_FACTORY_PATH, "examples")
LLAMA_FACTORY_EVALUATION_PATH = os.path.join(LLAMA_FACTORY_PATH, "evaluation")

# Source Code Paths
SRC_PATH = os.path.join(ROOT_PATH, "src")
SRC_LLAMAFACTORY_PATH = os.path.join(SRC_PATH, "llamafactory")

# Scripts Paths
SCRIPTS_PATH = os.path.join(ROOT_PATH, "scripts")

# Common File Extensions
YAML_EXT = ".yaml"
YML_EXT = ".yml"
JSON_EXT = ".json"
PY_EXT = ".py"
TXT_EXT = ".txt"
CSV_EXT = ".csv"
PKL_EXT = ".pkl"
PT_EXT = ".pt"
PTH_EXT = ".pth"

# Model File Extensions
MODEL_EXTENSIONS = [PT_EXT, PTH_EXT, "bin", "safetensors"]

# Default Configuration Files
DEFAULT_CONFIG_FILE = "configs/predenergy_config.yaml"

# Default Paths
DEFAULT_OUTPUT_PATH = "logs/Predenergy"
DEFAULT_MODEL_PATH = "checkpoints"
DEFAULT_DATA_PATH = "data"

# Training Constants
DEFAULT_SEED = 9899
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 1
DEFAULT_SEQ_LEN = 96
DEFAULT_PRED_LEN = 24

# Precision Types
PRECISION_BF16 = "bf16"
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"

# Feature Types
FEATURE_UNIVARIATE = "S"
FEATURE_MULTIVARIATE = "M"

# Normalization Types
NORM_NONE = 0
NORM_STANDARD = 1
NORM_MINMAX = 2

# Connection Types
CONNECTION_LINEAR = "linear"
CONNECTION_ATTENTION = "attention"
CONNECTION_CONCAT = "concat"
CONNECTION_ADAPTIVE = "adaptive"

# Data Loader Types
LOADER_STANDARD = "standard"
LOADER_UNIVERSAL = "universal"
LOADER_FORECASTING = "forecasting"

# Optimizer Types
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAMW = "adamw"
OPTIMIZER_SGD = "sgd"

# Scheduler Types
SCHEDULER_COSINE = "cosine"
SCHEDULER_LINEAR = "linear"
SCHEDULER_STEP = "step"

# Loss Functions
LOSS_MSE = "mse"
LOSS_MAE = "mae"
LOSS_HUBER = "huber"
LOSS_SMOOTH_L1 = "smooth_l1"

# Evaluation Strategies
EVAL_STRATEGY_EPOCH = "epoch"
EVAL_STRATEGY_STEPS = "steps"
EVAL_STRATEGY_NO = "no"

# Save Strategies
SAVE_STRATEGY_EPOCH = "epoch"
SAVE_STRATEGY_STEPS = "steps"
SAVE_STRATEGY_NO = "no"

# Frequency Types
FREQ_HOURLY = "h"
FREQ_DAILY = "d"
FREQ_WEEKLY = "w"
FREQ_MONTHLY = "m"
FREQ_YEARLY = "y"

# Activation Functions
ACTIVATION_GELU = "gelu"
ACTIVATION_RELU = "relu"
ACTIVATION_SILU = "silu"
ACTIVATION_TANH = "tanh"

# Model Types
MODEL_TYPE_STDM = "stdm"
MODEL_TYPE_MOTSE = "motse"
MODEL_TYPE_COMBINED = "combined"

# Data Processing Constants
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1

# Logging Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File Naming Patterns
MODEL_CHECKPOINT_PATTERN = "checkpoint-{}"
BEST_MODEL_NAME = "best_model"
LATEST_MODEL_NAME = "latest_model"

# Cache and Temporary Directories
CACHE_DIR = os.path.join(ROOT_PATH, ".cache")
TEMP_DIR = os.path.join(ROOT_PATH, ".temp")
LOGS_DIR = os.path.join(ROOT_PATH, "logs")
CHECKPOINTS_DIR = os.path.join(ROOT_PATH, "checkpoints")

# Ensure important directories exist
for directory in [CACHE_DIR, TEMP_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

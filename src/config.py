"""Central configuration for the fraud-detection pipeline.

Every random process in the project draws its seed from here, so a single
``SEED`` value makes the whole run (data generation, splits, resampling,
model training, bootstrap) reproducible bit-for-bit on the same library
versions.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Synthetic data generation (see src/data_generation.py for the model)
# --------------------------------------------------------------------------
N_CUSTOMERS = 3_000
N_DAYS = 90
TARGET_FRAUD_RATE = 0.01          # ~1% of transactions are fraudulent

# --------------------------------------------------------------------------
# Chronological split (fractions of the time-ordered transaction stream).
# Training on the past and testing on the future mimics deployment and
# prevents look-ahead leakage.
# --------------------------------------------------------------------------
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20                   # used only for decision-threshold tuning
# TEST_FRAC is the remaining 0.20

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
N_BOOTSTRAP = 1_000               # bootstrap resamples for confidence intervals
SAMPLING_RATIOS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]   # resampling ablation
ROBUSTNESS_SEEDS = [1, 2, 3, 4, 5]                    # extra synthetic "worlds"

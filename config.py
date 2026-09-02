from pathlib import Path


DATA_ROOT = Path("data")
RESULTS_DIR = Path("results")

TEST_PERSON = "P04"

FEATURE_MODE = "all"

RANDOM_STATE = 42

RF_N_ESTIMATORS = 200

MLP_HIDDEN_LAYERS = (64, 32)
MLP_MAX_ITER = 1000

SVM_KERNEL = "rbf"
SVM_C = 1.0
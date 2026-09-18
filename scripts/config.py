"""Global configuration: single source of truth for the random seed and paths."""
import logging
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for d in (DATA_PROCESSED, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to results/pipeline.log (append) and stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(RESULTS / "pipeline.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def log_versions(logger: logging.Logger) -> None:
    import platform
    import sklearn
    import scipy
    import pandas
    import numpy
    import xgboost
    import shap
    import matplotlib

    logger.info("=== Environment / reproducibility record ===")
    logger.info("Python: %s", platform.python_version())
    logger.info("SEED (numpy, sklearn splitters, xgboost): %d", SEED)
    logger.info("pandas=%s numpy=%s scikit-learn=%s scipy=%s", pandas.__version__, numpy.__version__, sklearn.__version__, scipy.__version__)
    logger.info("xgboost=%s shap=%s matplotlib=%s", xgboost.__version__, shap.__version__, matplotlib.__version__)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import scripts.train_linear as train_linear
import scripts.train_forest as train_forest
import scripts.train_catboost as train_catboost

if __name__ == '__main__':
    train_linear.start()
    train_forest.start()
    train_catboost.start()
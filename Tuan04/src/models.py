import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
K_FEATURES = 50

PARAMS_FILE = Path(__file__).with_name("hyperparams.json")

DEFAULT_PARAMS = {
    "SVR": {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
    "RandomForest": {"n_estimators": 300, "max_features": 0.5,
                     "min_samples_leaf": 1},
    "XGBoost": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3,
                "subsample": 0.8, "colsample_bytree": 0.6},
}

def load_params(path=PARAMS_FILE):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_PARAMS


def make_models(n_jobs=1, params=None):
    params = params or load_params()
    return {
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", **params["SVR"])),
        ]),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=n_jobs, **params["RandomForest"]),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_STATE,
            n_jobs=n_jobs, **params["XGBoost"]),
    }

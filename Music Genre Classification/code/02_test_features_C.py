
import common
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier

train, test, _ = common.load_data()

for version in ["v0", "v1", "v2"]:
    print(f"\n===== FEATURES {version} =====")

    X, y, X_test, ids, classes = build_features(
        train, test, version=version
    )

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        random_state=common.SEED
    )

    result = common.cross_validate_model(
        model,
        X,
        y,
        X_test,
        name=f"histgb_{version}_C"
    )

    print(result.summary())
    common.save_oof(result)

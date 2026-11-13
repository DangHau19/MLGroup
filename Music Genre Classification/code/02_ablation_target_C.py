
import numpy as np
import pandas as pd

import common
from features import build_features
from artist_target_encoder_C import ArtistTargetEncoder

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


def make_inner_oof(artists, y, seed=42):
    """Tao target encoding OOF cho tap train cua outer fold."""
    y = np.asarray(y, dtype=int)
    artists = artists.reset_index(drop=True)
    classes = np.unique(y)
    columns = [f"artist_te_{c}" for c in classes]

    result = pd.DataFrame(
        np.zeros((len(y), len(classes))),
        columns=columns
    )

    inner_cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed
    )

    for tr, va in inner_cv.split(artists, y):
        encoder = ArtistTargetEncoder(smoothing=20.0)
        encoder.fit(artists.iloc[tr], y[tr])

        encoded = encoder.transform(artists.iloc[va])
        result.iloc[va] = encoded[columns].to_numpy()

    return result


def main():
    train, test, _ = common.load_data()

    X, y, X_test, test_ids, classes = build_features(
        train, test, version="v2"
    )

    artists = train[["Artist Name"]].reset_index(drop=True)
    test_artists = test[["Artist Name"]].reset_index(drop=True)

    n_classes = len(classes)

    oof = np.zeros((len(y), n_classes))
    test_proba = np.zeros((len(test), n_classes))

    scores = []
    macro_scores = []

    for fold, (tr, va) in enumerate(common.get_folds(y), 1):
        print(f"\n===== FOLD {fold}/5 =====")

        # Target Encoding OOF ben trong train fold
        te_train = make_inner_oof(
            artists.iloc[tr].reset_index(drop=True),
            y[tr],
            seed=common.SEED + fold
        )

        # Fit encoder chi tren outer train fold
        encoder = ArtistTargetEncoder(smoothing=20.0)
        encoder.fit(artists.iloc[tr], y[tr])

        te_val = encoder.transform(artists.iloc[va])
        te_test = encoder.transform(test_artists)

        # Ghep cac dac trung
        X_tr = pd.concat(
            [X.iloc[tr].reset_index(drop=True), te_train],
            axis=1
        )

        X_va = pd.concat(
            [X.iloc[va].reset_index(drop=True),
             te_val.reset_index(drop=True)],
            axis=1
        )

        X_te = pd.concat(
            [X_test.reset_index(drop=True),
             te_test.reset_index(drop=True)],
            axis=1
        )

        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            random_state=common.SEED
        )

        model.fit(X_tr, y[tr])

        pred_proba = model.predict_proba(X_va)
        oof[va] = pred_proba

        pred = pred_proba.argmax(axis=1)
        metrics = common.score(y[va], pred)

        scores.append(metrics["accuracy"])
        macro_scores.append(metrics["macro_f1"])

        test_proba += model.predict_proba(X_te) / common.N_FOLDS

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")

    print("\n===== KET QUA CUOI =====")
    print(f"CV Accuracy: {np.mean(scores):.4f}")
    print(f"CV Macro F1: {np.mean(macro_scores):.4f}")

    result = common.CVResult(
        name="histgb_v2_artist_te_C",
        oof_proba=oof,
        test_proba=test_proba,
        fold_scores=scores,
        fold_macro_f1=macro_scores
    )

    common.save_oof(result)
    print("Da luu OOF va test probabilities.")


if __name__ == "__main__":
    main()

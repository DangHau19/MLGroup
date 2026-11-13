
import numpy as np
import pandas as pd

import common

from features import build_features
from artist_target_encoder_C import ArtistTargetEncoder

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

SEED = common.SEED
N_FOLDS = common.N_FOLDS


def create_model():
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        random_state=SEED
    )

# STEP 2: FOLD-SAFE FEATURE PREPROCESSING

def prepare_fold_features(train_fold, val_fold, test_data, version):
    """
    Build features using only training-fold statistics.
    """

    train_fold = train_fold.reset_index(drop=True)
    val_fold = val_fold.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Build features with the existing group implementation
    combined_test = pd.concat(
        [val_fold, test_data],
        ignore_index=True
    )

    X_train, y_train, X_combined, _, classes = build_features(
        train_fold,
        combined_test,
        version=version
    )

    n_val = len(val_fold)

    X_val = X_combined.iloc[:n_val].copy()
    X_test = X_combined.iloc[n_val:].copy()

    # Fit imputer using training fold only
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_val = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    return X_train, X_val, X_test



# STEP 3: NESTED OOF TARGET ENCODING

def create_inner_oof(artists, y, seed=42):
    artists = artists.reset_index(drop=True)
    y = np.asarray(y, dtype=int)

    columns = [f"artist_te_{i}" for i in range(11)]

    result = pd.DataFrame(
        np.zeros((len(y), 11)),
        columns=columns
    )

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed
    )

    for tr, va in cv.split(artists, y):
        encoder = ArtistTargetEncoder(smoothing=20.0)
        encoder.fit(artists.iloc[tr], y[tr])

        encoded = encoder.transform(artists.iloc[va])
        result.iloc[va] = encoded[columns].to_numpy()

    return result


# STEP 4: RUN FOLD-SAFE CROSS VALIDATION

def main():
    train, test, _ = common.load_data()

    y, classes = common.encode_labels(train["Class"])
    artists = train[["Artist Name"]].reset_index(drop=True)
    test_artists = test[["Artist Name"]].reset_index(drop=True)

    results = []

    for version in ["v0", "v1", "v2", "v2+TE"]:

        print(f"\n===== {version} =====", flush=True)

        scores = []
        macro_scores = []
        oof = np.zeros((len(y), len(classes)))
        test_proba = np.zeros((len(test), len(classes)))

        for fold, (tr, va) in enumerate(common.get_folds(y), 1):

            X_tr, X_va, X_te = prepare_fold_features(
                train.iloc[tr],
                train.iloc[va],
                test,
                "v2" if version == "v2+TE" else version
            )

            if version == "v2+TE":

                te_train = create_inner_oof(
                    artists.iloc[tr].reset_index(drop=True),
                    y[tr],
                    seed=SEED + fold
                )

                encoder = ArtistTargetEncoder(smoothing=20.0)
                encoder.fit(artists.iloc[tr], y[tr])

                te_val = encoder.transform(
                    artists.iloc[va]
                ).reset_index(drop=True)

                te_test = encoder.transform(
                    test_artists
                ).reset_index(drop=True)

                X_tr = pd.concat(
                    [X_tr.reset_index(drop=True), te_train],
                    axis=1
                )

                X_va = pd.concat(
                    [X_va.reset_index(drop=True), te_val],
                    axis=1
                )

                X_te = pd.concat(
                    [X_te.reset_index(drop=True), te_test],
                    axis=1
                )

            model = create_model()
            model.fit(X_tr, y[tr])

            proba = model.predict_proba(X_va)
            oof[va] = proba

            metrics = common.score(y[va], proba.argmax(axis=1))
            scores.append(metrics["accuracy"])
            macro_scores.append(metrics["macro_f1"])

            test_proba += model.predict_proba(X_te) / N_FOLDS

            print(
                f"Fold {fold}/5: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Macro F1={metrics['macro_f1']:.4f}",
                flush=True
            )

        result = common.CVResult(
            name=f"histgb_foldsafe_{version.replace('+', '_')}_C",
            oof_proba=oof,
            test_proba=test_proba,
            fold_scores=scores,
            fold_macro_f1=macro_scores
        )

        common.save_oof(result)

        results.append({
            "Version": version,
            "Accuracy": np.mean(scores),
            "Accuracy Std": np.std(scores),
            "Macro F1": np.mean(macro_scores)
        })

    result_df = pd.DataFrame(results)

    print("\n===== FINAL ABLATION =====")
    print(result_df.to_string(index=False))

    output = common.OOF_DIR / "ablation_foldsafe_C.csv"
    result_df.to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()

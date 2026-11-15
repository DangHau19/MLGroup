
import numpy as np
import pandas as pd
import common

from artist_target_encoder_C import ArtistTargetEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Inner OOF for unseen artists must use only each inner training fold's prior.
toy_y = np.array([0] * 7 + [1] * 5)
toy_X = pd.DataFrame({
    "Artist Name": [f"artist_{i}" for i in range(len(toy_y))]
})
toy_encoder = ArtistTargetEncoder(n_splits=4, random_state=common.SEED)
toy_oof = toy_encoder.fit_transform(toy_X, toy_y)
toy_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=common.SEED)

for inner_train, inner_valid in toy_cv.split(toy_X, toy_y):
    inner_prior = np.array([
        np.mean(toy_y[inner_train] == cls) for cls in toy_encoder.classes_
    ])
    fold_values = toy_oof.iloc[inner_valid][
        [f"artist_te_{cls}" for cls in toy_encoder.classes_]
    ].to_numpy()
    assert np.allclose(fold_values, np.tile(inner_prior, (len(inner_valid), 1)))

print("INNER OOF TARGET ENCODING TEST PASSED")

# The shared CV helper must preserve DataFrame column names for Pipeline steps.
cv_y = np.array([0] * 20 + [1] * 20)
cv_X = pd.DataFrame({
    "Artist Name": [f"artist_{i}" for i in range(len(cv_y))]
})
cv_test = pd.DataFrame({"Artist Name": ["unseen_artist", "artist_0"]})
pipeline = Pipeline([
    ("artist_target_encoder", ArtistTargetEncoder(
        smoothing=20.0,
        n_splits=4,
        random_state=common.SEED
    )),
    ("classifier", DummyClassifier(strategy="prior"))
])
pipeline_result = common.cross_validate_model(
    pipeline,
    cv_X,
    cv_y,
    cv_test,
    name="artist_te_pipeline_C",
    verbose=False
)
assert pipeline_result.oof_proba.shape == (len(cv_y), 2)
assert pipeline_result.test_proba.shape == (len(cv_test), 2)
assert np.isfinite(pipeline_result.oof_proba).all()
assert np.allclose(pipeline_result.oof_proba.sum(axis=1), 1.0)
assert np.allclose(pipeline_result.test_proba.sum(axis=1), 1.0)

print("PIPELINE/CV INTEGRATION TEST PASSED")

# Doc du lieu Kaggle
train, test, _ = common.load_data()

# Ma hoa nhan
y, classes = common.encode_labels(train["Class"])

# Chia train / validation
tr_idx, val_idx = common.get_folds(y)[0]

X_train = train.iloc[tr_idx][["Artist Name"]]
X_val = train.iloc[val_idx][["Artist Name"]]
X_test = test[["Artist Name"]]

# Chi fit tren train fold
encoder = ArtistTargetEncoder(smoothing=20.0)
encoder.fit(X_train, y[tr_idx])

# Ma hoa validation va test
val_encoded = encoder.transform(X_val)
test_encoded = encoder.transform(X_test)

print("So lop:", len(classes))
print("Validation:", val_encoded.shape)
print("Test:", test_encoded.shape)
print("NaN validation:", val_encoded.isna().sum().sum())
print("NaN test:", test_encoded.isna().sum().sum())

print("\n5 dong Target Encoding:")
print(val_encoded.head())

# Kiem tra tong xac suat bang 1
assert np.allclose(val_encoded.sum(axis=1), 1.0)
assert np.allclose(test_encoded.sum(axis=1), 1.0)

print("\nTARGET ENCODING TEST PASSED")

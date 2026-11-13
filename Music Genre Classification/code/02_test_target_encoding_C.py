
import numpy as np
import common

from artist_target_encoder_C import ArtistTargetEncoder

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

"""Check custom backprop by finite differences, and learning on known data."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp_D import SmallMLPClassifier


def main():
    rng = np.random.default_rng(42)
    X, y = rng.normal(size=(8, 3)), np.arange(8) % 2
    m = SmallMLPClassifier(hidden_layers=(4,), dropout=0)
    m.rng_ = rng
    m.weights_ = [rng.normal(size=(3, 4)), rng.normal(size=(4, 2))]
    m.biases_ = [rng.normal(size=4), rng.normal(size=2)]
    gw, gb = m._gradients(X, y)
    def loss():
        return -np.log(m._forward(X)[0][np.arange(len(y)), y]).mean()
    for param, grad in zip(m.weights_ + m.biases_, gw + gb):
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + 1e-6
            plus = loss()
            param[idx] = original - 1e-6
            minus = loss()
            param[idx] = original
            np.testing.assert_allclose(grad[idx], (plus - minus) / 2e-6, atol=1e-6)
    X, y = make_classification(n_samples=1000, n_features=6, n_informative=4,
                               n_redundant=0, class_sep=2, random_state=42)
    tr, va = train_test_split(np.arange(len(y)), stratify=y, random_state=42)
    m = SmallMLPClassifier(hidden_layers=(32,), dropout=0.1, max_epochs=70).fit(X[tr], y[tr])
    accuracy = (m.predict(X[va]) == y[va]).mean()
    assert accuracy > 0.9, accuracy
    p = m.predict_proba(X[va])
    np.testing.assert_array_equal(p, m.predict_proba(X[va]))
    np.testing.assert_allclose(p.sum(1), 1)
    assert len(m.validation_loss_) <= 70 and m.best_epoch_ <= m.n_iter_
    print(f"PASS: finite-difference gradients, dropout training, deterministic inference, accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()

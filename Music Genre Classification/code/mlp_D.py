"""Small NumPy MLP with inverted dropout, AdamW and an inner holdout.

This avoids an additional torch runtime. No outer validation data enter fit.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class SmallMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layers=(64, 32), dropout=0.15, weight_decay=0.001,
                 learning_rate=0.001, max_epochs=80, batch_size=256, patience=10,
                 random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _forward(self, X, training=False):
        activations, gates = [X], []
        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            z = activations[-1] @ W + b
            gate = (z > 0).astype(float)
            if training and self.dropout:
                gate *= (self.rng_.random(z.shape) >= self.dropout) / (1 - self.dropout)
            gates.append(gate)
            activations.append(np.maximum(z, 0) * gate)
        logits = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True), activations, gates

    def _gradients(self, X, y):
        p, activations, gates = self._forward(X, training=True)
        delta = p.copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        gw, gb = [None] * len(self.weights_), [None] * len(self.biases_)
        for layer in reversed(range(len(self.weights_))):
            gw[layer] = activations[layer].T @ delta
            gb[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights_[layer].T) * gates[layer - 1]
        return gw, gb

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, targets = np.unique(y, return_inverse=True)
        self.rng_ = np.random.default_rng(self.random_state)
        widths = (X.shape[1],) + tuple(self.hidden_layers) + (len(self.classes_),)
        self.weights_ = [self.rng_.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(widths[:-1], widths[1:])]
        self.biases_ = [np.zeros(b) for b in widths[1:]]
        tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=targets,
                                 random_state=self.random_state)
        params = self.weights_ + self.biases_
        moment1, moment2 = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best_loss, stale, step = np.inf, 0, 0
        best = [p.copy() for p in params]
        self.validation_loss_ = []
        for epoch in range(self.max_epochs):
            order = self.rng_.permutation(tr)
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                gw, gb = self._gradients(X[idx], targets[idx])
                gradients = gw + gb
                norm = np.sqrt(sum(np.sum(g * g) for g in gradients))
                step += 1
                for j, (p, g) in enumerate(zip(params, gradients)):
                    g = g * min(1, 5 / max(norm, 1e-12))
                    moment1[j] = 0.9 * moment1[j] + 0.1 * g
                    moment2[j] = 0.999 * moment2[j] + 0.001 * g * g
                    # Decoupled weight decay only on weight matrices.
                    if j < len(self.weights_):
                        p *= 1 - self.learning_rate * self.weight_decay
                    p -= self.learning_rate * (moment1[j] / (1 - 0.9 ** step)) / (np.sqrt(moment2[j] / (1 - 0.999 ** step)) + 1e-8)
            p = self._forward(X[va])[0]
            loss = -np.log(np.maximum(p[np.arange(len(va)), targets[va]], 1e-12)).mean()
            self.validation_loss_.append(float(loss))
            if loss < best_loss - 1e-4:
                best_loss, stale = loss, 0
                self.best_epoch_ = epoch + 1
                best = [p.copy() for p in params]
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.n_iter_ = epoch + 1
        for p, saved in zip(params, best):
            p[:] = saved
        return self

    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float))[0]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

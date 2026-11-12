"""Feed-forward neural network (PyTorch) wrapped as a scikit-learn classifier."""
import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Dense ReLU network with dropout, Adam, mini-batches and early stopping.

    A stratified slice of the training data (validation_fraction) is held out for
    early stopping; the weights from the epoch with the lowest validation loss are kept.
    The per-epoch loss/accuracy history is stored in `history_`.
    """

    def __init__(self, hidden=(32, 16), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=32,
                 max_epochs=300, patience=25, validation_fraction=0.2, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _network(self, n_inputs):
        layers, size = [], n_inputs
        for units in self.hidden:
            layers += [nn.Linear(size, units), nn.ReLU(), nn.Dropout(self.dropout)]
            size = units
        layers.append(nn.Linear(size, 1))  # one logit; sigmoid is applied in the loss / predict_proba
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)  # tiny network: avoids thread over-subscription inside parallel CV
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=self.validation_fraction,
                                                    stratify=y, random_state=self.random_state)
        X_tr, y_tr = torch.tensor(X_tr), torch.tensor(y_tr).reshape(-1, 1)
        X_val, y_val = torch.tensor(X_val), torch.tensor(y_val).reshape(-1, 1)

        self.model_ = self._network(X.shape[1])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.random_state)
        best_loss, best_state, stale, self.history_ = np.inf, None, 0, []

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            for idx in torch.randperm(len(X_tr), generator=generator).split(self.batch_size):
                optimizer.zero_grad()
                loss_fn(self.model_(X_tr[idx]), y_tr[idx]).backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                out_tr, out_val = self.model_(X_tr), self.model_(X_val)
                row = {"epoch": epoch,
                       "train_loss": loss_fn(out_tr, y_tr).item(),
                       "validation_loss": loss_fn(out_val, y_val).item(),
                       "train_accuracy": ((out_tr > 0).float() == y_tr).float().mean().item(),
                       "validation_accuracy": ((out_val > 0).float() == y_val).float().mean().item()}
            self.history_.append(row)
            if row["validation_loss"] < best_loss - 1e-4:
                best_loss, best_state, stale = row["validation_loss"], copy.deepcopy(self.model_.state_dict()), 0
                self.best_epoch_ = epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.tensor(np.asarray(X, dtype="float32")))).numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

"""Linear probe training for corporate identity detection in model activations.

Trains logistic regression probes on intermediate activations to determine
whether and where models encode corporate identity information.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Optional

from research.config import ExperimentConfig, experiment_config


class CorporateIdentityProbe:
    """Trains and evaluates linear probes for corporate identity detection.

    Probes are trained on model activations extracted at each layer to determine
    where identity information is linearly separable in the residual stream.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or experiment_config

    def prepare_data(
        self, activations: dict, layer: int
    ) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """Extract features at a specific layer from the activations dict.

        Parameters
        ----------
        activations : dict
            Nested dict: activations[identity][query] = tensor(num_layers, hidden_dim)
        layer : int
            Layer index to extract features from.

        Returns
        -------
        X : np.ndarray of shape (n_samples, hidden_dim)
        y : np.ndarray of shape (n_samples,) — integer-encoded labels
        label_encoder : LabelEncoder fitted on the identity labels
        """
        X_parts = []
        y_parts = []

        for identity, queries in activations.items():
            for query, tensor in queries.items():
                # tensor shape: (num_layers, hidden_dim)
                # Explicit float32 cast handles bfloat16 tensors (numpy has no bfloat16)
                if hasattr(tensor, "numpy"):
                    feature = tensor[layer].float().numpy()
                elif hasattr(tensor, "cpu"):
                    feature = tensor[layer].cpu().float().numpy()
                else:
                    feature = np.asarray(tensor[layer], dtype=np.float32)
                X_parts.append(feature)
                y_parts.append(identity)

        X = np.stack(X_parts, axis=0)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_parts)
        return X, y, label_encoder

    def train_binary_probe(
        self, X_pos: np.ndarray, X_neg: np.ndarray
    ) -> dict:
        """Train a binary logistic regression probe with stratified cross-validation.

        Parameters
        ----------
        X_pos : np.ndarray of shape (n_pos, hidden_dim)
            Positive class activations.
        X_neg : np.ndarray of shape (n_neg, hidden_dim)
            Negative class activations.

        Returns
        -------
        dict with keys: model, auroc, accuracy, f1, cv_scores, direction
        """
        X = np.concatenate([X_pos, X_neg], axis=0)
        y = np.concatenate([
            np.ones(len(X_pos), dtype=int),
            np.zeros(len(X_neg), dtype=int),
        ])

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Train/val split first, then CV on training set only
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Use LogisticRegressionCV to tune regularization strength via CV
        probe = LogisticRegressionCV(
            Cs=self.config.probe_C_values,
            cv=cv,
            max_iter=1000,
            random_state=self.config.random_state,
            solver="lbfgs",
            scoring="accuracy",
        )
        probe.fit(X_train, y_train)

        # CV scores on training split only (avoids data leakage)
        cv_scores = cross_val_score(
            LogisticRegression(
                C=probe.C_[0], max_iter=1000,
                random_state=self.config.random_state, solver="lbfgs",
            ),
            X_train, y_train, cv=cv, scoring="accuracy",
        )

        # Held-out metrics
        y_val_pred = probe.predict(X_val)
        y_val_proba = probe.predict_proba(X_val)[:, 1]
        val_accuracy = float(accuracy_score(y_val, y_val_pred))
        val_auroc = float(roc_auc_score(y_val, y_val_proba))
        val_f1 = float(f1_score(y_val, y_val_pred))

        # Train metrics (for overfitting detection)
        y_train_pred = probe.predict(X_train)
        y_train_proba = probe.predict_proba(X_train)[:, 1]
        train_accuracy = float(accuracy_score(y_train, y_train_pred))
        train_auroc = float(roc_auc_score(y_train, y_train_proba))

        direction = probe.coef_[0].copy()
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        return {
            "model": probe,
            "auroc": val_auroc,
            "accuracy": val_accuracy,
            "f1": val_f1,
            "train_accuracy": train_accuracy,
            "train_auroc": train_auroc,
            "overfit_gap": train_accuracy - val_accuracy,
            "cv_scores": cv_scores.tolist(),
            "best_C": float(probe.C_[0]),
            "direction": direction,
        }

    def train_multiclass_probe(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        """Train a one-vs-rest multiclass logistic regression probe.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, hidden_dim)
        y : np.ndarray of shape (n_samples,) — integer-encoded labels

        Returns
        -------
        dict with keys: model, accuracy, f1_macro, confusion_matrix, cv_scores
        """
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Train/val split first, then CV on training set only
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Use LogisticRegressionCV to tune regularization strength via CV
        probe = LogisticRegressionCV(
            Cs=self.config.probe_C_values,
            cv=cv,
            max_iter=1000,
                        random_state=self.config.random_state,
            solver="lbfgs",
            scoring="accuracy",
        )
        probe.fit(X_train, y_train)

        # CV scores on training split only (avoids data leakage)
        cv_scores = cross_val_score(
            LogisticRegression(
                C=probe.C_[0], max_iter=1000,                 random_state=self.config.random_state, solver="lbfgs",
            ),
            X_train, y_train, cv=cv, scoring="accuracy",
        )

        # Held-out metrics
        y_val_pred = probe.predict(X_val)
        val_accuracy = float(accuracy_score(y_val, y_val_pred))
        val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

        # Train metrics
        y_train_pred = probe.predict(X_train)
        train_accuracy = float(accuracy_score(y_train, y_train_pred))

        return {
            "model": probe,
            "accuracy": val_accuracy,
            "f1_macro": val_f1,
            "train_accuracy": train_accuracy,
            "overfit_gap": train_accuracy - val_accuracy,
            "confusion_matrix": confusion_matrix(y_val, y_val_pred),
            "cv_scores": cv_scores.tolist(),
            "best_C": float(probe.C_[0]),
        }

    def layer_sweep(
        self,
        activations: dict,
        probe_type: str = "binary",
        identity_pair: Optional[tuple] = None,
    ) -> dict:
        """Train probes at every layer and return per-layer results.

        Parameters
        ----------
        activations : dict
            activations[identity][query] = tensor(num_layers, hidden_dim)
        probe_type : str
            "binary" or "multiclass".
        identity_pair : tuple of (pos_identity, neg_identity), optional
            Required when probe_type is "binary". Specifies which two
            identities to contrast.

        Returns
        -------
        dict mapping layer_idx (int) -> probe results dict
        """
        # Determine number of layers from first activation tensor
        first_identity = next(iter(activations))
        first_query = next(iter(activations[first_identity]))
        sample = activations[first_identity][first_query]
        if hasattr(sample, "shape"):
            num_layers = sample.shape[0]
        else:
            num_layers = len(sample)

        results = {}
        for layer in range(num_layers):
            if probe_type == "binary":
                if identity_pair is None:
                    raise ValueError(
                        "identity_pair must be provided for binary probing"
                    )
                pos_id, neg_id = identity_pair

                # Gather activations for the two identities at this layer
                X_pos_parts = []
                for query, tensor in activations[pos_id].items():
                    if hasattr(tensor, "numpy"):
                        X_pos_parts.append(tensor[layer].float().numpy())
                    elif hasattr(tensor, "cpu"):
                        X_pos_parts.append(tensor[layer].cpu().float().numpy())
                    else:
                        X_pos_parts.append(np.asarray(tensor[layer], dtype=np.float32))

                X_neg_parts = []
                for query, tensor in activations[neg_id].items():
                    if hasattr(tensor, "numpy"):
                        X_neg_parts.append(tensor[layer].float().numpy())
                    elif hasattr(tensor, "cpu"):
                        X_neg_parts.append(tensor[layer].cpu().float().numpy())
                    else:
                        X_neg_parts.append(np.asarray(tensor[layer], dtype=np.float32))

                X_pos = np.stack(X_pos_parts, axis=0)
                X_neg = np.stack(X_neg_parts, axis=0)
                results[layer] = self.train_binary_probe(X_pos, X_neg)
            else:
                X, y, label_encoder = self.prepare_data(activations, layer)
                result = self.train_multiclass_probe(X, y)
                result["label_encoder"] = label_encoder
                results[layer] = result

        return results

    def get_identity_direction(self, probe_results: dict) -> np.ndarray:
        """Extract the normalized direction vector from a trained binary probe.

        Parameters
        ----------
        probe_results : dict
            Results dict returned by train_binary_probe (must contain 'direction'
            or 'model' key).

        Returns
        -------
        np.ndarray — unit-length direction vector in activation space
        """
        if "direction" in probe_results:
            return probe_results["direction"]
        model = probe_results["model"]
        direction = model.coef_[0].copy()
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        return direction

    def train_random_baseline(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        """Evaluate a random-weights baseline (no learning).

        Projects activations onto a random Gaussian direction and thresholds
        at zero. This estimates chance-level performance for comparison.
        Uses the same train/val split as trained probes for consistency.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, hidden_dim)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        dict with keys: accuracy, f1, auroc (all expected ~chance)
        """
        # Use same train/val split as other probes for consistent comparison
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        rng = np.random.RandomState(self.config.random_state)
        random_direction = rng.randn(X.shape[1])
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-12)

        # Evaluate on held-out set only
        scores = X_val @ random_direction
        y_pred = (scores > 0).astype(int)

        # Normalise scores to [0, 1] for AUROC
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            auroc = float(roc_auc_score(y_val, scores_norm))
        else:
            auroc = None

        return {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1": float(f1_score(y_val, y_pred, average="binary" if len(unique_labels) == 2 else "macro")),
            "auroc": auroc,
            "direction": random_direction,
        }

    def train_surface_baseline(
        self, tokenized_inputs: list, y: np.ndarray
    ) -> dict:
        """Train a probe on raw tokenized input features (bag-of-tokens).

        This baseline checks whether the probe is simply detecting superficial
        input artifacts rather than learned internal representations.
        Uses LogisticRegressionCV for consistent regularization tuning with
        the main probes.

        Parameters
        ----------
        tokenized_inputs : list
            List of token-id sequences (list of lists/arrays).
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        dict with keys: model, accuracy, f1, cv_scores, best_C
        """
        # Build bag-of-tokens feature matrix
        max_token_id = max(
            token_id
            for seq in tokenized_inputs
            for token_id in seq
        ) + 1
        # Cap vocabulary dimension for memory efficiency
        vocab_size = min(max_token_id, 50_000)

        X_bow = np.zeros((len(tokenized_inputs), vocab_size), dtype=np.float32)
        for i, seq in enumerate(tokenized_inputs):
            for token_id in seq:
                if token_id < vocab_size:
                    X_bow[i, token_id] += 1.0

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Train/val split first, then CV on training set only (consistent with other probes)
        X_train, X_val, y_train, y_val = train_test_split(
            X_bow, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Use LogisticRegressionCV for regularization tuning (consistent with main probes)
        probe = LogisticRegressionCV(
            Cs=self.config.probe_C_values,
            cv=cv,
            max_iter=1000,
            random_state=self.config.random_state,
            solver="lbfgs",
            scoring="accuracy",
        )
        probe.fit(X_train, y_train)

        # CV scores on training split with best C
        cv_scores = cross_val_score(
            LogisticRegression(
                C=probe.C_[0], max_iter=1000,
                random_state=self.config.random_state, solver="lbfgs",
            ),
            X_train, y_train, cv=cv, scoring="accuracy",
        )

        y_val_pred = probe.predict(X_val)
        y_train_pred = probe.predict(X_train)

        unique_labels = np.unique(y)
        f1_avg = "binary" if len(unique_labels) == 2 else "macro"

        val_accuracy = float(accuracy_score(y_val, y_val_pred))
        train_accuracy = float(accuracy_score(y_train, y_train_pred))

        return {
            "model": probe,
            "accuracy": val_accuracy,
            "train_accuracy": train_accuracy,
            "overfit_gap": train_accuracy - val_accuracy,
            "f1": float(f1_score(y_val, y_val_pred, average=f1_avg)),
            "cv_scores": cv_scores.tolist(),
            "best_C": float(probe.C_[0]),
        }

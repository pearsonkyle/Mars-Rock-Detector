import os
import logging
import numpy as np
import pandas as pd
import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay

logger = logging.getLogger(__name__)


class Ensembler():
    def __init__(self, models: list):
        """
        Ensemble of classifiers that train on randomised splits of the data
        and predict by averaging all member models.

        Parameters
        ----------
        models:
            List of scikit-learn–compatible classifier instances that expose
            ``fit``, ``predict``, and ``predict_proba`` methods.
        """
        self.models = models
        self.acc = []  # validation accuracy per model
        self._do_preprocess: bool = True  # mirrors the preprocess flag passed to fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Per-sample normalisation: subtract sample mean, divide by (1 + std)."""
        return (X - X.mean(axis=1, keepdims=True)) / (1 + X.std(axis=1, keepdims=True))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, preprocess: bool = True, n_folds: int = None) -> None:
        """
        Train every model in the ensemble using stratified k-fold cross-validation.

        Each model is assigned to a fold in round-robin order.  The model trains
        on the remaining (k-1) folds and its validation accuracy is measured on
        the held-out fold.  Using ``StratifiedKFold`` ensures each fold preserves
        the class-label distribution of the full dataset.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        y:
            Label vector of shape (n_samples,).
        preprocess:
            When True (default) each sample is normalised before training.
        n_folds:
            Number of folds for stratified k-fold cross-validation.  Defaults
            to the number of models in the ensemble so that every model is
            trained on a distinct held-out fold.
        """
        self.xdim = X.shape[1]
        self._do_preprocess = preprocess

        Xp = self._preprocess(X) if preprocess else X

        # Build stratified k-fold splits (preserves class balance per fold)
        k = n_folds if n_folds is not None else len(self.models)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        folds = list(skf.split(Xp, y))

        for i, model in enumerate(self.models):

            # cycle through folds in round-robin if n_models > n_folds
            train_idx, val_idx = folds[i % k]
            X_train, X_val = Xp[train_idx], Xp[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # train the model
            model.fit(X_train, y_train)

            # compute validation accuracy on the held-out fold
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            self.acc.append(acc)
            logger.debug("Model %d (fold %d/%d) validation accuracy: %.4f", i, i % k + 1, k, acc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, prob: bool = False) -> tuple:
        """
        Predict with every model and return the mean and std across the ensemble.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        prob:
            When True return class probabilities instead of hard labels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_prediction, std_prediction) both of the same shape.
        """
        Xp = self._preprocess(X) if self._do_preprocess else X
        if prob:
            predictions = np.array([model.predict_proba(Xp) for model in self.models])
        else:
            predictions = np.array([model.predict(Xp) for model in self.models])
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def predict_best(self, X: np.ndarray, prob: bool = False) -> np.ndarray:
        """
        Predict using only the highest-accuracy model in the ensemble.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        prob:
            When True return class probabilities instead of hard labels.

        Returns
        -------
        np.ndarray
            Predictions from the best model.
        """
        Xp = self._preprocess(X) if self._do_preprocess else X
        if prob:
            predictions = self.best.predict_proba(Xp)
        else:
            predictions = self.best.predict(Xp)
        return predictions

    @property
    def best(self):
        """Return the model with the highest validation accuracy."""
        return self.models[np.argmax(self.acc)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the ensemble to *path* using joblib.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``ensemble.pkl``).
        """
        joblib.dump(self, path)
        logger.info("Ensemble saved to %s", path)
        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Ensembler":
        """
        Load a previously saved ensemble from *path*.

        Parameters
        ----------
        path:
            Source file path produced by :meth:`save`.

        Returns
        -------
        Ensembler
            The deserialised ensemble.
        """
        ensemble = joblib.load(path)
        logger.info("Ensemble loaded from %s", path)
        print(f"Ensemble loaded from {path}")
        return ensemble

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_roc(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot the ROC curve for every model in the ensemble.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        y:
            True binary labels of shape (n_samples,).
        """
        Xp = self._preprocess(X) if self._do_preprocess else X

        # overplot the ROC curve for each model in the ensemble
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, model in enumerate(self.models):
            y_score = model.predict_proba(Xp)
            RocCurveDisplay.from_predictions(
                y, y_score[:, 1],
                name=f"Model {i}",
                color=plt.cm.jet(i / len(self.models)),
                alpha=0.7,
                ax=ax,
            )

        ax.grid(True, ls='--')
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        ax.axis("square")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC curves for each model in the ensemble")
        plt.legend()
        plt.tight_layout()
        plt.show()


def train_ensembler(
    train_file: str,
    test_file: str,
    class_key: str = 'label',
    plot: bool = False,
    n_models: int = 9,
    n_jobs: int = -1,
    n_folds: int = None,
) -> Ensembler:
    """
    Build and train an ensemble of Random Forest classifiers.

    Parameters
    ----------
    train_file:
        Path to the CSV file containing training samples.
    test_file:
        Path to the CSV file containing test samples (used for ROC plotting).
    class_key:
        Name of the column that holds the class labels.
    plot:
        When True, display ROC curves after training.
    n_models:
        Number of Random Forest models in the ensemble (default: 9).
    n_jobs:
        Number of parallel jobs forwarded to each ``RandomForestClassifier``.
        ``-1`` uses all available CPU cores.
    n_folds:
        Number of folds for stratified k-fold cross-validation passed to
        :meth:`Ensembler.fit`.  Defaults to ``n_models`` so each model is
        evaluated on a distinct held-out fold.

    Returns
    -------
    Ensembler
        The trained ensemble.
    """
    # Load a dataset in a Pandas dataframe.
    rock_df = pd.read_csv(train_file)

    # randomize the order of the data
    rock_df = rock_df.sample(frac=1).reset_index(drop=True)

    # split into labels and data arrays
    rock_label = rock_df[class_key].values.astype(np.float32)
    rock_data = rock_df.drop(class_key, axis=1).values

    logger.info(
        "Training on %d images with %d features each...",
        rock_data.shape[0], rock_data.shape[1],
    )
    print(f"Training on {rock_data.shape[0]} images with {rock_data.shape[1]} features each...")

    # create an ensemble of models
    models = Ensembler([
        RandomForestClassifier(class_weight='balanced', n_jobs=n_jobs, random_state=i)
        for i in range(n_models)
    ])

    # train the ensemble
    models.fit(rock_data, rock_label, n_folds=n_folds)

    # print accuracy of each model
    print(models.acc)

    # roc plot
    test_df = pd.read_csv(test_file)
    test_label = test_df[class_key].values.astype(np.float32)
    test_data = test_df.drop(class_key, axis=1).values

    if plot:
        models.plot_roc(test_data, test_label)

    return models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Random Forest ensemble for Mars rock detection."
    )

    parser.add_argument("-ws", "--windowsize", type=int, default=11,
            help="size of training sample output in px")  # size of training data

    parser.add_argument("-nm", "--n_models", type=int, default=9,
            help="number of models in the ensemble")

    parser.add_argument("-k", "--n_folds", type=int, default=None,
            help="number of folds for stratified k-fold cross-validation "
                 "(default: same as --n_models)")

    parser.add_argument("-j", "--n_jobs", type=int, default=-1,
            help="number of parallel jobs for each Random Forest (-1 = all CPUs)")

    parser.add_argument("-s", "--save", type=str, default=None,
            help="path to save the trained ensemble (e.g. ensemble.pkl)")

    parser.add_argument("-l", "--load", type=str, default=None,
            help="path to load a previously saved ensemble instead of training")

    parser.add_argument("-p", "--plot", action="store_true",
            help="plot ROC curves after training")

    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()

    if args.load:
        ensemble = Ensembler.load(args.load)
    else:
        ensemble = train_ensembler(
            f'training/training_data_{args.windowsize**2}.csv',
            f'training/testing_data_{args.windowsize**2}.csv',
            n_models=args.n_models,
            n_jobs=args.n_jobs,
            n_folds=args.n_folds,
            plot=args.plot,
        )

        if args.save:
            ensemble.save(args.save)

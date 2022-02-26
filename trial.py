import logging
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from datasets import load_all
from moolearn.neural_network._multilayer_perceptron import (
    MLPClassifier as MOMLPClassifier,
)

RESULTS_FULL_PATH = Path(__file__).parent / "results_full"
RESULTS_SUMMARY_PATH = Path(__file__).parent / "results_summary"


class OVACrossEntropyLoss(_Loss):
    def __init__(self, cls: int):
        super().__init__()
        self.cls = cls

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ova_input = torch.stack(
            [
                input[:, : self.cls].sum(dim=1) + input[:, (self.cls + 1) :].sum(dim=1),
                input[:, self.cls],
            ],
            dim=1,
        )
        ova_target = (target == self.cls).long()

        return F.cross_entropy(ova_input, ova_target)


if __name__ == "__main__":
    simplefilter("ignore", category=ConvergenceWarning)

    datasets = load_all()

    rows = []

    for dataset_name, folds in datasets.items():
        logging.info(f"Evaluating {dataset_name}...")

        for i, ((X_train, y_train), (X_test, y_test)) in enumerate(folds):
            n_classes = len(np.unique(y_test))

            clf = MLPClassifier()
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            bac = balanced_accuracy_score(y_test, pred)

            rows.append([dataset_name, i, "MLP", acc, bac])

            objectives = [OVACrossEntropyLoss(cls=cls) for cls in range(n_classes)]
            clf = MOMLPClassifier(objectives)
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test).mean(axis=0)
            pred = proba.argmax(axis=1)
            acc = accuracy_score(y_test, pred)
            bac = balanced_accuracy_score(y_test, pred)

            rows.append([dataset_name, i, "MO-MLP (OVA)", acc, bac])

    df_full = pd.DataFrame(
        rows, columns=["Dataset", "Fold", "Classifier", "Acc", "BAC"]
    )
    df_full = (
        df_full.groupby(["Dataset", "Classifier"])[["Acc", "BAC"]]
        .agg("mean")
        .reset_index()
    )

    df_summary = (
        df_full.groupby(["Classifier"])[["Acc", "BAC"]].agg("mean").reset_index()
    )

    for path in [RESULTS_FULL_PATH, RESULTS_SUMMARY_PATH]:
        path.mkdir(exist_ok=True, parents=True)

    df_full.to_csv(RESULTS_FULL_PATH / "test.csv")
    df_summary.to_csv(RESULTS_SUMMARY_PATH / "test.csv")

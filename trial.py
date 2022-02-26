import argparse
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
    logging.basicConfig(level=logging.INFO)
    simplefilter("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser()

    parser.add_argument("-max_iter", type=int)
    parser.add_argument("-n_layers", type=int)
    parser.add_argument("-n_test_rays", type=int)

    args = parser.parse_args()

    datasets = load_all()

    rows = []

    for dataset_name, folds in datasets.items():
        logging.info(f"Evaluating {dataset_name}...")

        for i, ((X_train, y_train), (X_test, y_test)) in enumerate(folds):
            n_classes = len(np.unique(y_test))

            clf = MLPClassifier(
                hidden_layer_sizes=(100,) * args.n_layers, max_iter=args.max_iter
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            bac = balanced_accuracy_score(y_test, pred)

            rows.append([dataset_name, i, "MLP", acc, bac])

            objectives = [OVACrossEntropyLoss(cls=cls) for cls in range(n_classes)]
            clf = MOMLPClassifier(
                objectives,
                hidden_layer_sizes=(100,) * args.n_layers,
                max_iter=args.max_iter,
                n_test_rays=args.n_test_rays,
            )
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

    trial_name = f"max_iter={args.max_iter}_n_layers={args.n_layers}_n_test_rays={args.n_test_rays}"

    df_full.to_csv(RESULTS_FULL_PATH / f"{trial_name}.csv", index=False)
    df_summary.to_csv(RESULTS_SUMMARY_PATH / f"{trial_name}.csv", index=False)

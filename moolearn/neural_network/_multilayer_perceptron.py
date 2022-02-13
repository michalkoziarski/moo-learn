from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes: Tuple = (100,),
        activation: nn.modules.Module = nn.ReLU,
        *,
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False,
        device: Union[torch.device, str] = "auto",
    ):
        self.hidden_layer_size = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        self.model = None

    def _initialize_model(self, n_features: int, n_classes: int) -> nn.modules.Module:
        layers = []

        for i in range(len(self.hidden_layer_size) + 1):
            if i == 0:
                input_dim = n_features
            else:
                input_dim = self.hidden_layer_size[i - 1]

            if i == len(self.hidden_layer_size):
                output_dim = n_classes
            else:
                output_dim = self.hidden_layer_size[i]

            layers.append(nn.Linear(input_dim, output_dim))

            if i < len(self.hidden_layer_size):
                layers.append(self.activation())

        return nn.Sequential(*layers)

    def _get_batch_size(self, n_samples: int, default: int = 200) -> int:
        if self.batch_size == "auto":
            batch_size = min(default, n_samples)
        else:
            batch_size = self.batch_size

        return batch_size

    def _get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = self.device

        return device

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPClassifier:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        n_samples = X.shape[0]

        device = self._get_device()

        self.model = self._initialize_model(n_features, n_classes)
        self.model.to(device)
        self.model.train()

        train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
        train_loader = DataLoader(
            train_dataset, batch_size=self._get_batch_size(n_samples), shuffle=True
        )

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.max_iter):
            if self.verbose:
                iterator = tqdm(train_loader)
            else:
                iterator = train_loader

            losses = []

            for batch in iterator:
                inputs, targets = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if self.verbose:
                    losses.append(loss.cpu().detach().numpy())

                    iterator.set_description(
                        f"Epoch {epoch + 1}/{self.max_iter}. Train loss: {loss:.4f}."
                    )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        self.model.eval()

        dataset = TensorDataset(torch.Tensor(X))
        loader = DataLoader(dataset, batch_size=self._get_batch_size(n_samples))

        predictions = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self._get_device())
                outputs = F.softmax(self.model(inputs), dim=1)

                for output in outputs:
                    predictions.append(output.cpu().detach().numpy())

        return np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

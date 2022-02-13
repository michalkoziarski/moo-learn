from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ._cosmos import circle_points


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        objectives: List[nn.modules.Module],
        hidden_layer_sizes: Tuple = (100,),
        activation: nn.modules.Module = nn.ReLU,
        *,
        alpha: float = 1.2,
        lambd: float = 2.0,
        n_test_rays: int = 25,
        weight_decay: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False,
        device: Union[torch.device, str] = "auto",
    ):
        self.objectives = objectives
        self.hidden_layer_size = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.lambd = lambd
        self.n_test_rays = n_test_rays
        self.weight_decay = weight_decay
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
                input_dim = n_features + len(self.objectives)
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
        if self.random_state is not None:
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
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.max_iter):
            if self.verbose:
                iterator = tqdm(train_loader)
            else:
                iterator = train_loader

            losses = {
                k: []
                for k in ["Total loss", "Cos. sim."]
                + [f"O{i + 1}" for i, _ in enumerate(self.objectives)]
            }

            for batch in iterator:
                optimizer.zero_grad()

                inputs, targets = batch[0].to(device), batch[1].to(device)

                alphas = torch.from_numpy(
                    np.random.dirichlet([self.alpha for _ in self.objectives], 1)
                    .astype(np.float32)
                    .flatten()
                ).to(device)

                inputs = torch.cat((inputs, alphas.repeat(inputs.shape[0], 1)), dim=1)
                outputs = self.model(inputs)

                total_loss = 0.0
                objective_losses = []

                for alpha, objective in zip(alphas, self.objectives):
                    objective_loss = objective(outputs, targets)
                    total_loss += alpha * objective_loss
                    objective_losses.append(objective_loss)

                cos_sim = F.cosine_similarity(
                    torch.stack(objective_losses), alphas, dim=0
                )

                total_loss -= self.lambd * cos_sim
                total_loss.backward()

                optimizer.step()

                if self.verbose:
                    losses["Total loss"].append(total_loss.cpu().detach().numpy())
                    losses["Cos. sim."].append(cos_sim.cpu().detach().numpy())

                    for i, l in enumerate(objective_losses):
                        losses[f"O{i + 1}"].append(l.cpu().detach().numpy())

                    loss_string = ", ".join(
                        [f"{k}: {np.mean(v):.4f}" for k, v in losses.items()]
                    )

                    iterator.set_description(
                        f"Epoch {epoch + 1}/{self.max_iter}. {loss_string}."
                    )

        return self

    def predict_proba(
        self, X: np.ndarray, test_rays: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n_samples = X.shape[0]
        device = self._get_device()

        self.model.eval()

        dataset = TensorDataset(torch.Tensor(X))
        loader = DataLoader(dataset, batch_size=self._get_batch_size(n_samples))

        if test_rays is None:
            test_rays = circle_points(self.n_test_rays, dim=len(self.objectives))

        predictions = [[] for _ in test_rays]

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)

                for i, ray in enumerate(test_rays):
                    ray = torch.from_numpy(ray.astype(np.float32)).to(device)
                    ray /= ray.sum()

                    x = torch.cat((inputs, ray.repeat(inputs.shape[0], 1)), dim=1)

                    outputs = F.softmax(self.model(x), dim=1)

                    for output in outputs:
                        predictions[i].append(output.cpu().detach().numpy())

        return np.array(predictions)

    def predict(
        self, X: np.ndarray, test_rays: Optional[np.ndarray] = None
    ) -> np.ndarray:
        probas = self.predict_proba(X, test_rays)

        return np.array([np.argmax(p, axis=1) for p in probas])

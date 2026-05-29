
"""
Réseau de neurones Feedforward pour prédire le score final de Faraway.

Architecture :
  Input (50) → Dense(128, ReLU) → Dropout(0.2)
             → Dense(64, ReLU)  → Dropout(0.2)
             → Dense(32, ReLU)
             → Dense(1, Linear)  ← score prédit

Entraînement : régression supervisée sur des parties simulées.
  - X : vecteur d'état (50 features) au moment où la carte est jouée
  - y : score final obtenu à la fin de cette partie
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from ai.neural.encoder import INPUT_DIM


class FarawayNet(nn.Module):
    """
    Réseau Feedforward pour prédire le score final.
    Entrée  : vecteur état+carte (INPUT_DIM = 50)
    Sortie  : score prédit (1 valeur)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            # Dropout seulement sur les premières couches
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # sortie scalaire

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, INPUT_DIM) → (batch, 1)"""
        return self.net(x)

    def predict(self, x: np.ndarray) -> float:
        """
        Prédit le score pour UN vecteur numpy.
        Usage : score = model.predict(state_vector)
        """
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            return float(self.forward(t).squeeze())

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les scores pour un batch de vecteurs.
        Usage : scores = model.predict_batch(batch_matrix)  → shape (N,)
        """
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return self.forward(t).squeeze(1).numpy()

    def save(self, path: str) -> None:
        torch.save({
            "state_dict":  self.state_dict(),
            "input_dim":   self.input_dim,
            "hidden_dims": self.hidden_dims,
        }, path)
        print(f"💾 Modèle sauvegardé : {path}")

    @classmethod
    def load(cls, path: str) -> "FarawayNet":
        data   = torch.load(path, map_location="cpu")
        model  = cls(
            input_dim   = data["input_dim"],
            hidden_dims = data["hidden_dims"],
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model


class FarawayTrainer:
    """
    Entraîne FarawayNet sur des données de parties simulées.

    Données attendues :
      X : np.ndarray (N, INPUT_DIM) — vecteurs d'état
      y : np.ndarray (N,)           — scores finaux correspondants
    """

    def __init__(
        self,
        model: FarawayNet,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        self.model      = model
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()
        self.batch_size = batch_size
        self.losses: list[float] = []

    # def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
    #     """Lance une époque d'entraînement. Retourne la perte moyenne."""
    #     self.model.train()
    #     X_t = torch.tensor(X, dtype=torch.float32)
    #     y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    #
    #     # Mélanger
    #     idx = torch.randperm(len(X_t))
    #     X_t, y_t = X_t[idx], y_t[idx]
    #
    #     total_loss = 0.0
    #     n_batches  = 0
    #
    #     for start in range(0, len(X_t), self.batch_size):
    #         xb = X_t[start:start + self.batch_size]
    #         yb = y_t[start:start + self.batch_size]
    #
    #         self.optimizer.zero_grad()
    #         pred = self.model(xb)
    #         loss = self.loss_fn(pred, yb)
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         total_loss += loss.item()
    #         n_batches  += 1
    #
    #     avg_loss = total_loss / max(n_batches, 1)
    #     self.losses.append(avg_loss)
    #     return avg_loss

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray | None = None,
            epochs: int = 50,
            val_split: float = 0.1,
            verbose: bool = True,
    ) -> None:
        """Entraîne sur `epochs` époques avec validation optionnelle."""

        n_val = int(len(X) * val_split)
        n_train = len(X) - n_val

        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        if weights is not None:
            w_train, w_val = weights[:n_train], weights[n_train:]
        else:
            w_train = w_val = None

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(X_train, y_train, w_train)
            self.losses.append(train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32)
                yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

                pred = self.model(Xv)

                if w_val is not None:
                    wv = torch.tensor(w_val, dtype=torch.float32).unsqueeze(1)
                    val_loss = (wv * (pred - yv) ** 2).mean().item()
                else:
                    val_loss = self.loss_fn(pred, yv).item()

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"train={train_loss:.3f} | val={val_loss:.3f}")

        if best_state:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"  ✅ Meilleur modèle restauré (val={best_val_loss:.3f})")

    def train_epoch(self, X, y, weights=None) -> float:
        self.model.train()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Poids par exemple — tours tardifs = poids élevé
        if weights is not None:
            w_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        else:
            w_t = torch.ones(len(X_t), 1)

        idx = torch.randperm(len(X_t))
        X_t, y_t, w_t = X_t[idx], y_t[idx], w_t[idx]

        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_t), self.batch_size):
            xb = X_t[start:start + self.batch_size]
            yb = y_t[start:start + self.batch_size]
            wb = w_t[start:start + self.batch_size]

            self.optimizer.zero_grad()
            pred = self.model(xb)

            # Loss pondérée : erreur × poids du tour
            loss = (wb * (pred - yb) ** 2).sum() / wb.sum()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


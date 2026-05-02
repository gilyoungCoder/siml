"""
Mode-Aware Gradient Model Utilities
Clustering and mode-aware guidance utilities for harmful latent management
"""

import os
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class ClusterManager:
    """
    Manages clustering of harmful latents for mode-aware guidance.

    Clusters harmful latents into K modes and provides utilities for:
    - Fitting clusters from collected latents
    - Assigning new latents to clusters
    - Computing distances to cluster centroids
    """

    def __init__(
        self,
        n_clusters: int = 10,
        pooling: str = "mean",
        random_state: int = 42,
    ):
        """
        Args:
            n_clusters: Number of clusters (modes)
            pooling: Pooling method for latent features ("mean", "flatten", "attention")
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.pooling = pooling
        self.random_state = random_state

        self.centroids: Optional[torch.Tensor] = None
        self.kmeans: Optional[KMeans] = None
        self._fitted = False

    def pool_features(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pool latent features for clustering.

        Args:
            latents: Latent tensor of shape (N, C, H, W)

        Returns:
            Pooled features of shape (N, D)
        """
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)

        N, C, H, W = latents.shape

        if self.pooling == "mean":
            # Global average pooling: (N, C, H, W) -> (N, C)
            features = latents.mean(dim=[2, 3])
        elif self.pooling == "flatten":
            # Flatten all spatial dimensions: (N, C, H, W) -> (N, C*H*W)
            features = latents.view(N, -1)
        elif self.pooling == "attention":
            # Attention-weighted pooling
            # Compute attention weights based on L2 norm
            attention = torch.norm(latents, dim=1, keepdim=True)  # (N, 1, H, W)
            attention = F.softmax(attention.view(N, -1), dim=-1).view(N, 1, H, W)
            features = (latents * attention).sum(dim=[2, 3])  # (N, C)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return features

    def fit(
        self,
        latents: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Fit clusters to latent data.

        Args:
            latents: Latent tensor of shape (N, C, H, W)
            save_path: Optional path to save centroids

        Returns:
            Cluster labels for each latent
        """
        # Pool features
        features = self.pool_features(latents)
        features_np = features.cpu().numpy()

        # Fit KMeans
        print(f"[ClusterManager] Fitting {self.n_clusters} clusters on {len(features_np)} samples...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(features_np)

        # Store centroids as torch tensor
        self.centroids = torch.from_numpy(self.kmeans.cluster_centers_).float()
        self._fitted = True

        # Save if path provided
        if save_path:
            self.save(save_path)

        return labels

    def predict(self, latents: torch.Tensor) -> np.ndarray:
        """
        Predict cluster assignments for new latents.

        Args:
            latents: Latent tensor of shape (N, C, H, W)

        Returns:
            Cluster labels
        """
        if not self._fitted:
            raise RuntimeError("ClusterManager not fitted. Call fit() first.")

        # Use distance-based prediction (works without kmeans object)
        distances = self.get_distances(latents)  # (N, K)
        labels = distances.argmin(dim=-1).cpu().numpy()

        return labels

    def get_distances(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from latents to all cluster centroids.

        Args:
            latents: Latent tensor of shape (N, C, H, W)

        Returns:
            Distance matrix of shape (N, K) where K is n_clusters
        """
        if not self._fitted:
            raise RuntimeError("ClusterManager not fitted. Call fit() first.")

        features = self.pool_features(latents)  # (N, D)
        centroids = self.centroids.to(features.device)  # (K, D)

        # Compute L2 distances
        # (N, 1, D) - (1, K, D) -> (N, K, D) -> (N, K)
        distances = torch.norm(
            features.unsqueeze(1) - centroids.unsqueeze(0),
            dim=-1
        )

        return distances

    def get_nearest_cluster(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get nearest cluster and distance for each latent.

        Args:
            latents: Latent tensor of shape (N, C, H, W)

        Returns:
            Tuple of (cluster_indices, distances) each of shape (N,)
        """
        distances = self.get_distances(latents)
        min_distances, cluster_indices = distances.min(dim=-1)

        return cluster_indices, min_distances

    def save(self, path: str):
        """Save cluster centroids and metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        save_dict = {
            "centroids": self.centroids,
            "n_clusters": self.n_clusters,
            "pooling": self.pooling,
        }
        torch.save(save_dict, path)
        print(f"[ClusterManager] Saved centroids to {path}")

    def load(self, path: str, timestep: Optional[int] = None):
        """Load cluster centroids and metadata.

        Args:
            path: Path to saved centroids
            timestep: For multi-timestep files, which timestep to load.
                     If None, uses the smallest timestep (cleanest).
        """
        save_dict = torch.load(path, map_location="cpu")

        # Handle multi-timestep format
        if "timesteps" in save_dict and isinstance(save_dict.get("centroids"), dict):
            available_timesteps = save_dict["timesteps"]
            centroids_dict = save_dict["centroids"]

            if timestep is None:
                # Use smallest timestep (cleanest latents)
                timestep = min(available_timesteps)

            if timestep not in centroids_dict:
                raise ValueError(f"Timestep {timestep} not found. Available: {available_timesteps}")

            self.centroids = centroids_dict[timestep]
            self.n_clusters = save_dict["n_clusters"]
            self.pooling = save_dict.get("pooling", "mean")
            self._fitted = True
            self._timesteps = available_timesteps
            self._centroids_dict = centroids_dict

            print(f"[ClusterManager] Loaded {self.n_clusters} centroids from {path} (t={timestep})")
            print(f"  Available timesteps: {available_timesteps}")
        else:
            # Single timestep format (backward compatible)
            self.centroids = save_dict["centroids"]
            self.n_clusters = save_dict["n_clusters"]
            self.pooling = save_dict.get("pooling", "mean")
            self._fitted = True

            print(f"[ClusterManager] Loaded {self.n_clusters} centroids from {path}")

    def set_timestep(self, timestep: int):
        """Switch to centroids for a different timestep (multi-timestep mode only)."""
        if not hasattr(self, '_centroids_dict'):
            raise RuntimeError("Not in multi-timestep mode. Load a multi-timestep file first.")

        if timestep not in self._centroids_dict:
            raise ValueError(f"Timestep {timestep} not found. Available: {self._timesteps}")

        self.centroids = self._centroids_dict[timestep]
        print(f"[ClusterManager] Switched to t={timestep}")


def fit_harmful_clusters(
    latents: torch.Tensor,
    n_clusters: int = 10,
    pooling: str = "mean",
    save_path: Optional[str] = None,
) -> Tuple[ClusterManager, np.ndarray]:
    """
    Convenience function to fit clusters on harmful latents.

    Args:
        latents: Harmful latent tensor of shape (N, C, H, W)
        n_clusters: Number of clusters
        pooling: Pooling method
        save_path: Optional path to save centroids

    Returns:
        Tuple of (ClusterManager, labels)
    """
    manager = ClusterManager(
        n_clusters=n_clusters,
        pooling=pooling,
    )
    labels = manager.fit(latents, save_path=save_path)

    return manager, labels

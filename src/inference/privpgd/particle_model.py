import pickle
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict
import logging

import pandas as pd
import torch
import numpy as np

from src.inference.dataset import Dataset
from src.inference.domain import Domain
from src.inference.embedding import Embedding

if TYPE_CHECKING:
    from src.inference.factor import Factor

logging.basicConfig(level=logging.INFO)


class AdaptiveParticleModel:
    """Enhanced ParticleModel with adaptive particle allocation and hierarchical structure."""

    def __init__(
            self,
            domain: "Domain",
            embedding: Optional["Embedding"] = None,
            n_particles: int = 500,
            data_init: Optional["Dataset"] = None,
            adaptive_mode: str = "none",  # "none", "variance", "importance", "entropy"
            use_hierarchical: bool = False,
            hierarchy_levels: int = 3,
            min_particles_per_region: int = 10,
    ):
        """
        Initializes an adaptive ParticleModel with intelligent particle allocation.

        Args:
            domain (Domain): The domain associated with the particle model.
            embedding (Optional[Embedding]): An embedding object.
            n_particles (int): Base number of particles.
            data_init (Optional[Dataset]): Dataset for initialization.
            adaptive_mode (str): Adaptive allocation strategy:
                - "none": Uniform allocation (baseline)
                - "variance": Allocate based on local variance
                - "importance": Allocate based on importance weights
                - "entropy": Allocate based on local entropy
            use_hierarchical (bool): Enable hierarchical multi-scale representation.
            hierarchy_levels (int): Number of hierarchy levels (for hierarchical mode).
            min_particles_per_region (int): Minimum particles per region.
        """
        self.domain = domain
        self.adaptive_mode = adaptive_mode
        self.use_hierarchical = use_hierarchical
        self.hierarchy_levels = hierarchy_levels
        self.min_particles_per_region = min_particles_per_region

        if embedding is None:
            self.embedding = Embedding(domain)
        else:
            self.embedding = embedding

        # Initialize base particles
        if not data_init:
            self.n_particles = n_particles
            self.X = torch.rand(
                n_particles,
                self.embedding.bijection_length,
                dtype=torch.float32,
                device=self.embedding.device,
            )
            self.w = torch.full(
                (n_particles,),
                1 / n_particles,
                dtype=torch.float32,
                device=self.embedding.device,
            )
        else:
            self.n_particles = data_init.df.shape[0]
            if data_init.weights is not None:
                self.w = torch.tensor(
                    data_init.weights / sum(data_init.weights),
                    dtype=torch.float32,
                    device=self.embedding.device,
                )
            else:
                self.w = torch.full(
                    (self.n_particles,),
                    1 / self.n_particles,
                    dtype=torch.float32,
                    device=self.embedding.device,
                )
            self.X = self.embedding.embedd(data_init.df)

        # Hierarchical structure
        self.hierarchical_particles = None
        if use_hierarchical:
            self._initialize_hierarchical_structure()

        # Adaptive allocation metadata
        self.region_importance = None
        self.particle_allocation_map = None

        logging.info(
            f"Initialized AdaptiveParticleModel: "
            f"mode={adaptive_mode}, hierarchical={use_hierarchical}, "
            f"n_particles={n_particles}"
        )

    def _initialize_hierarchical_structure(self):
        """Initialize hierarchical multi-scale particle representation."""
        self.hierarchical_particles = []

        # Create particles at different scales
        for level in range(self.hierarchy_levels):
            # Coarser levels have fewer particles
            scale_factor = 2 ** level
            n_particles_level = max(
                self.min_particles_per_region,
                self.n_particles // scale_factor
            )

            # Cluster particles into regions
            if level == 0:
                # Finest level: use original particles
                particles_level = self.X.clone()
                weights_level = self.w.clone()
            else:
                # Coarser levels: cluster and aggregate
                particles_level, weights_level = self._cluster_particles(
                    self.X, self.w, n_particles_level
                )

            self.hierarchical_particles.append({
                'level': level,
                'particles': particles_level,
                'weights': weights_level,
                'n_particles': n_particles_level,
            })

        logging.info(
            f"Initialized {self.hierarchy_levels} hierarchical levels: "
            f"{[h['n_particles'] for h in self.hierarchical_particles]}"
        )

    def _cluster_particles(
            self,
            particles: torch.Tensor,
            weights: torch.Tensor,
            n_clusters: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster particles using k-means for hierarchical representation.

        Args:
            particles: Particle positions
            weights: Particle weights
            n_clusters: Target number of clusters

        Returns:
            Clustered particles and their weights
        """
        # Simple k-means clustering
        n_particles = particles.shape[0]

        if n_particles <= n_clusters:
            return particles.clone(), weights.clone()

        # Initialize cluster centers randomly
        indices = torch.randperm(n_particles)[:n_clusters]
        centers = particles[indices].clone()

        # K-means iterations
        for _ in range(10):
            # Assign particles to nearest center
            distances = torch.cdist(particles, centers)
            assignments = torch.argmin(distances, dim=1)

            # Update centers
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    # Weighted average
                    cluster_weights = weights[mask]
                    cluster_particles = particles[mask]
                    centers[k] = torch.sum(
                        cluster_particles * cluster_weights.unsqueeze(1),
                        dim=0
                    ) / cluster_weights.sum()

        # Compute cluster weights
        cluster_weights = torch.zeros(n_clusters, device=self.embedding.device)
        for k in range(n_clusters):
            mask = assignments == k
            cluster_weights[k] = weights[mask].sum()

        # Normalize
        cluster_weights = cluster_weights / cluster_weights.sum()

        return centers, cluster_weights

    def compute_region_importance(
            self,
            marginals: Optional[List[Tuple[str, ...]]] = None,
            data_reference: Optional["Dataset"] = None
    ) -> torch.Tensor:
        """
        Compute importance of different regions for adaptive allocation.

        Args:
            marginals: List of marginal queries to prioritize
            data_reference: Reference dataset for computing statistics

        Returns:
            Importance scores for each particle
        """
        if self.adaptive_mode == "none":
            return torch.ones(self.n_particles, device=self.embedding.device)

        # Initialize importance scores
        importance = torch.zeros(self.n_particles, device=self.embedding.device)

        if self.adaptive_mode == "variance":
            # Allocate more particles to high-variance regions
            importance = self._compute_variance_importance()

        elif self.adaptive_mode == "entropy":
            # Allocate more particles to high-entropy regions
            importance = self._compute_entropy_importance()

        elif self.adaptive_mode == "importance":
            # Combined importance metric
            var_imp = self._compute_variance_importance()
            ent_imp = self._compute_entropy_importance()
            importance = 0.5 * var_imp + 0.5 * ent_imp

        # Normalize
        importance = importance / importance.sum()

        self.region_importance = importance
        return importance

    def _compute_variance_importance(self) -> torch.Tensor:
        """Compute variance-based importance for each particle."""
        # Compute local variance around each particle
        n_neighbors = min(50, self.n_particles // 10)

        # For each particle, find neighbors and compute local variance
        distances = torch.cdist(self.X, self.X)

        importance = torch.zeros(self.n_particles, device=self.embedding.device)
        for i in range(self.n_particles):
            # Find k nearest neighbors
            _, indices = torch.topk(distances[i], k=n_neighbors + 1, largest=False)
            neighbors = self.X[indices[1:]]  # Exclude self

            # Compute local variance
            local_var = torch.var(neighbors, dim=0).sum()
            importance[i] = local_var

        return importance

    def _compute_entropy_importance(self) -> torch.Tensor:
        """Compute entropy-based importance for each particle."""
        # Estimate local density using kernel density estimation
        bandwidth = 0.1
        n_particles = self.n_particles

        # Compute pairwise distances
        distances = torch.cdist(self.X, self.X)

        # Gaussian kernel
        kernel_values = torch.exp(-distances ** 2 / (2 * bandwidth ** 2))

        # Density estimation
        density = kernel_values.sum(dim=1) / n_particles

        # Entropy = -log(density)
        # Higher entropy (lower density) = higher importance
        importance = -torch.log(density + 1e-10)

        return importance

    def adaptive_reallocate_particles(
            self,
            target_n_particles: Optional[int] = None,
            importance_threshold: float = 0.5,
    ):
        """
        Adaptively reallocate particles based on importance scores.

        Args:
            target_n_particles: Target total number of particles (None = keep current)
            importance_threshold: Only reallocate if importance > threshold
        """
        if self.region_importance is None:
            self.compute_region_importance()

        if target_n_particles is None:
            target_n_particles = self.n_particles

        # Identify high-importance regions
        high_importance_mask = self.region_importance > importance_threshold
        n_high_importance = high_importance_mask.sum().item()

        if n_high_importance == 0:
            logging.warning("No high-importance regions found, skipping reallocation")
            return

        # Calculate how many particles to allocate to each region
        base_allocation = target_n_particles // self.n_particles
        extra_particles = target_n_particles - (base_allocation * self.n_particles)

        # Allocate extra particles proportionally to importance
        importance_sorted_indices = torch.argsort(
            self.region_importance, descending=True
        )

        new_particles = []
        new_weights = []

        for i in range(self.n_particles):
            idx = i
            n_particles_for_region = base_allocation

            # Allocate extra particles to high-importance regions
            if i < extra_particles:
                n_particles_for_region += int(
                    extra_particles * self.region_importance[importance_sorted_indices[i]]
                )

            # Generate new particles around this region
            if n_particles_for_region > 0:
                # Add jitter to create diversity
                jitter = torch.randn(
                    n_particles_for_region,
                    self.embedding.bijection_length,
                    device=self.embedding.device
                ) * 0.05

                region_particles = self.X[idx].unsqueeze(0).repeat(
                    n_particles_for_region, 1
                ) + jitter

                # Clamp to [0, 1]
                region_particles = torch.clamp(region_particles, 0, 1)

                region_weights = torch.full(
                    (n_particles_for_region,),
                    self.w[idx] / n_particles_for_region,
                    device=self.embedding.device
                )

                new_particles.append(region_particles)
                new_weights.append(region_weights)

        # Concatenate and update
        self.X = torch.cat(new_particles, dim=0)
        self.w = torch.cat(new_weights, dim=0)
        self.w = self.w / self.w.sum()  # Renormalize
        self.n_particles = self.X.shape[0]

        logging.info(f"Reallocated particles: {self.n_particles} total particles")

    def get_particles_at_level(self, level: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get particles at a specific hierarchy level.

        Args:
            level: Hierarchy level (0 = finest, higher = coarser)

        Returns:
            Particles and weights at the specified level
        """
        if not self.use_hierarchical:
            return self.X, self.w

        if level >= len(self.hierarchical_particles):
            level = len(self.hierarchical_particles) - 1

        return (
            self.hierarchical_particles[level]['particles'],
            self.hierarchical_particles[level]['weights']
        )

    def refine_hierarchical_particles(
            self,
            level: int,
            refinement_factor: int = 2
    ):
        """
        Refine particles at a specific level by splitting them.

        Args:
            level: Level to refine
            refinement_factor: How many particles to create from each parent
        """
        if not self.use_hierarchical or level >= len(self.hierarchical_particles):
            return

        particles = self.hierarchical_particles[level]['particles']
        weights = self.hierarchical_particles[level]['weights']

        # Split each particle into multiple refined particles
        refined_particles = []
        refined_weights = []

        for i in range(particles.shape[0]):
            # Create refined particles with small jitter
            jitter = torch.randn(
                refinement_factor,
                self.embedding.bijection_length,
                device=self.embedding.device
            ) * 0.02

            refined = particles[i].unsqueeze(0).repeat(refinement_factor, 1) + jitter
            refined = torch.clamp(refined, 0, 1)

            refined_particles.append(refined)
            refined_weights.append(
                torch.full(
                    (refinement_factor,),
                    weights[i] / refinement_factor,
                    device=self.embedding.device
                )
            )

        # Update the level
        self.hierarchical_particles[level]['particles'] = torch.cat(refined_particles)
        self.hierarchical_particles[level]['weights'] = torch.cat(refined_weights)
        self.hierarchical_particles[level]['n_particles'] *= refinement_factor

    @staticmethod
    def save(model: "AdaptiveParticleModel", path: str) -> None:
        """Save the AdaptiveParticleModel to a file."""
        pickle.dump(model, open(path, "wb"))

    @staticmethod
    def load(path: str) -> "AdaptiveParticleModel":
        """Load an AdaptiveParticleModel from a file."""
        return pickle.load(open(path, "rb"))

    @staticmethod
    def base_norm(v: torch.Tensor) -> torch.Tensor:
        """Compute the L2 norm of a tensor."""
        return torch.norm(v, p=2, dim=-1)

    def compute_distance_matrix(
            self,
            attrs: Optional[List[str]] = None,
            other: Optional["AdaptiveParticleModel"] = None,
            use_level: int = 0,
    ) -> torch.Tensor:
        """
        Compute distance matrix between particles.

        Args:
            attrs: Subset of attributes to consider
            other: Another model to compare with
            use_level: Hierarchy level to use (if hierarchical)
        """
        if attrs is None:
            attrs = self.domain.attrs
        if other is None:
            other = self

        # Get particles at specified level
        X_self, _ = self.get_particles_at_level(use_level)
        X_other, _ = other.get_particles_at_level(use_level)

        return self.embedding.compute_distance_matrix(attrs, X_self, X_other)

    def project(
            self,
            attrs: Union[List[str], Tuple[str, ...]],
            use_level: int = 0
    ) -> "Factor":
        """
        Project the particle model onto a subset of attributes.

        Args:
            attrs: Attributes to project onto
            use_level: Hierarchy level to use
        """
        attrs_shape = [self.domain.config[attr] for attr in attrs]
        newDomain = Domain(attrs, attrs_shape)

        X, w = self.get_particles_at_level(use_level)
        return self.embedding.project(newDomain, X, w)

    def datavector(self, flatten: bool = True, use_level: int = 0) -> torch.Tensor:
        """
        Materialize distribution as a data vector.

        Args:
            flatten: Whether to flatten the result
            use_level: Hierarchy level to use
        """
        return self.project(self.domain.attrs, use_level).datavector()

    def synthetic_data(
            self,
            records: Optional[int] = None,
            use_adaptive_sampling: bool = True,
            use_level: int = 0
    ) -> "Dataset":
        """
        Generate synthetic tabular data from the particle model.

        Args:
            records: Number of records (None or 0 = use all particles)
            use_adaptive_sampling: Use importance-weighted sampling
            use_level: Hierarchy level to use
        """
        X, w = self.get_particles_at_level(use_level)

        # Handle None or 0 records
        if records is None or records == 0:
            records = X.shape[0]

        if use_adaptive_sampling and records > 0 and records < X.shape[0]:
            # Sample particles based on weights
            indices = torch.multinomial(
                w,
                num_samples=min(records, X.shape[0]),
                replacement=True
            )
            X_sampled = X[indices]
            w_sampled = w[indices]
            w_sampled = w_sampled / w_sampled.sum()
        else:
            # Use all particles or when not using adaptive sampling
            if records >= X.shape[0]:
                # Use all particles if requested more than available
                X_sampled = X
                w_sampled = w
            else:
                # Take first 'records' particles
                X_sampled = X[:records]
                w_sampled = w[:records]
                w_sampled = w_sampled / w_sampled.sum()

        X_disc = self.embedding.discretize(X_sampled, self.domain.attrs)
        weights = w_sampled.detach().cpu().numpy()
        df = pd.DataFrame(
            X_disc.detach().cpu().numpy(), columns=self.domain.attrs
        )
        return Dataset(df, self.domain, weights=weights)


# Backward compatibility: alias to original name
class ParticleModel(AdaptiveParticleModel):
    """Backward compatible ParticleModel (uses AdaptiveParticleModel with default settings)."""

    def __init__(
            self,
            domain: "Domain",
            embedding: Optional["Embedding"] = None,
            n_particles: int = 500,
            data_init: Optional["Dataset"] = None,
    ):
        super().__init__(
            domain=domain,
            embedding=embedding,
            n_particles=n_particles,
            data_init=data_init,
            adaptive_mode="none",
            use_hierarchical=False,
        )

    def synthetic_data(self, records: Optional[int] = None) -> "Dataset":
        """
        Generate synthetic tabular data (backward compatible signature).

        Args:
            records: Number of records (None = use all particles)

        Returns:
            Generated synthetic dataset
        """
        return super().synthetic_data(
            records=records,
            use_adaptive_sampling=False,
            use_level=0
        )
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict

import numpy as np

from src.mechanisms.ektelo_matrix import Identity
from src.mechanisms.mechanism import Mechanism
from src.mechanisms.privacy_calibrator import gaussian_mech

if TYPE_CHECKING:
    from src.inference.dataset import Dataset
    from src.inference.pgm.inference import FactoredInference
    from src.inference.privpgd.inference import AdvancedSlicedInference


class KWayEnhanced(Mechanism):
    def __init__(
            self,
            epsilon: float,
            delta: float = 0.00001,
            degree: int = 2,
            bounded: bool = True,
            prng: np.random = np.random,
            weight_strategy: str = "uniform",
            custom_weights: Optional[Dict[Tuple[str, ...], float]] = None,
            noise_type: str = "gaussian",  # "gaussian", "renyi", "gp"
            rdp_orders: Optional[np.ndarray] = None,
            gp_lengthscale: float = 1.0,
            gp_variance: float = 1.0,
    ):
        """
        Enhanced K-Way mechanism with weighted privacy budget and advanced noise mechanisms.

        Args:
            epsilon (float): Total differential privacy budget.
            delta (float): DP parameter delta.
            degree (int): Degree of K-Way mechanism.
            bounded (bool): Bounded vs unbounded DP.
            prng (np.random): Random number generator.
            weight_strategy (str): Budget allocation strategy.
            custom_weights (Optional[Dict]): Custom weights.
            noise_type (str): Noise mechanism - "gaussian", "renyi", or "gp"
            rdp_orders (Optional[np.ndarray]): RDP orders for Rényi DP.
            gp_lengthscale (float): Lengthscale for GP kernel.
            gp_variance (float): Variance for GP kernel.
        """
        super(KWayEnhanced, self).__init__(
            epsilon=epsilon,
            delta=delta,
            bounded=bounded,
            prng=prng,
            noise_type=noise_type,
            rdp_orders=rdp_orders
        )
        self.k = degree
        self.weight_strategy = weight_strategy
        self.custom_weights = custom_weights or {}

        # Set GP parameters (these are already set in parent __init__ but we override them)
        self.gp_lengthscale = gp_lengthscale
        self.gp_variance = gp_variance

        logging.info(f"Initialized KWayEnhanced with noise_type='{noise_type}'")

    def _compute_variance_weights(
            self, data: "Dataset", workload: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], float]:
        """Compute variance-based weights."""
        weights = {}
        variances = []

        for cl in workload:
            x = data.project(cl).datavector()
            var = np.var(x)
            variances.append(var)
            weights[cl] = var

        total_var = sum(variances)
        if total_var > 0:
            weights = {k: v / total_var for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(workload) for k in workload}

        return weights

    def _compute_entropy_weights(
            self, data: "Dataset", workload: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], float]:
        """Compute entropy-based weights."""
        weights = {}
        entropies = []

        for cl in workload:
            x = data.project(cl).datavector()
            probs = x / (x.sum() + 1e-10)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
            weights[cl] = entropy

        total_entropy = sum(entropies)
        if total_entropy > 0:
            weights = {k: v / total_entropy for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(workload) for k in workload}

        return weights

    def _compute_importance_weights(
            self, data: "Dataset", workload: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], float]:
        """Compute hybrid importance weights."""
        var_weights = self._compute_variance_weights(data, workload)
        ent_weights = self._compute_entropy_weights(data, workload)

        weights = {
            k: 0.5 * var_weights[k] + 0.5 * ent_weights[k]
            for k in workload
        }

        return weights

    def _get_weights(
            self, data: "Dataset", workload: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], float]:
        """Get weights based on strategy."""
        if self.weight_strategy == "uniform":
            return {cl: 1.0 / len(workload) for cl in workload}
        elif self.weight_strategy == "variance":
            return self._compute_variance_weights(data, workload)
        elif self.weight_strategy == "entropy":
            return self._compute_entropy_weights(data, workload)
        elif self.weight_strategy == "importance":
            return self._compute_importance_weights(data, workload)
        elif self.weight_strategy == "custom":
            total_weight = sum(self.custom_weights.values())
            return {k: v / total_weight for k, v in self.custom_weights.items()}
        else:
            logging.warning(f"Unknown strategy '{self.weight_strategy}'. Using uniform.")
            return {cl: 1.0 / len(workload) for cl in workload}

    def _allocate_noise_gaussian(
            self, weights: Dict, workload: List, epsilon: float, delta: float
    ) -> Dict[Tuple[str, ...], float]:
        """Allocate noise using standard Gaussian mechanism."""
        try:
            sigma_base = gaussian_mech(epsilon, delta, k=len(workload))["sigma"]
        except ValueError:
            logging.warning("Using conservative noise estimate.")
            sigma_base = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        sigma_allocations = {}
        num_queries = len(workload)

        for cl in workload:
            weight = weights[cl]
            weight_normalized = weight * num_queries
            sigma_allocations[cl] = sigma_base / np.sqrt(weight_normalized)

        return sigma_allocations

    def _allocate_noise_renyi(
            self, weights: Dict, workload: List, epsilon: float, delta: float
    ) -> Dict[Tuple[str, ...], float]:
        """Allocate noise using Rényi DP accounting."""
        # Calibrate base sigma using RDP
        sigma_base = self.calibrate_gaussian_rdp(
            sensitivity=1.0,
            epsilon=epsilon,
            delta=delta
        )

        sigma_allocations = {}
        num_queries = len(workload)

        for cl in workload:
            weight = weights[cl]
            weight_normalized = weight * num_queries
            sigma_allocations[cl] = sigma_base / np.sqrt(weight_normalized)

        logging.info(f"RDP calibrated sigma_base: {sigma_base:.4f}")
        return sigma_allocations

    def _create_query_embedding(
            self, workload: List[Tuple[str, ...]], data: "Dataset"
    ) -> np.ndarray:
        """
        Create embedding for queries to use with GP-DP.
        Embeds each marginal query into a feature space.
        """
        # Create a simple embedding based on attribute indices
        attr_to_idx = {attr: idx for idx, attr in enumerate(data.domain.attrs)}

        embeddings = []
        for cl in workload:
            # Create a binary vector indicating which attributes are in this query
            emb = np.zeros(len(data.domain.attrs))
            for attr in cl:
                if attr in attr_to_idx:
                    emb[attr_to_idx[attr]] = 1.0
            embeddings.append(emb)

        return np.array(embeddings)

    def _allocate_noise_gp(
            self,
            weights: Dict,
            workload: List,
            epsilon: float,
            delta: float,
            data: "Dataset"
    ) -> Tuple[Dict, np.ndarray]:
        """
        Allocate noise using Gaussian Process DP.
        Returns both sigma allocations and query embeddings.
        """
        # Create query embeddings for GP kernel
        query_locations = self._create_query_embedding(workload, data)

        # Compute GP covariance matrix
        cov_matrix = self.gp_noise_covariance(
            locations=query_locations,
            epsilon=epsilon,
            delta=delta,
            sensitivity=1.0
        )

        # Extract per-query noise scales (diagonal of covariance)
        sigma_allocations = {}
        for idx, cl in enumerate(workload):
            # Base sigma from diagonal
            sigma_base = np.sqrt(cov_matrix[idx, idx])

            # Apply weight-based scaling
            weight = weights[cl]
            weight_normalized = weight * len(workload)
            sigma_allocations[cl] = sigma_base / np.sqrt(weight_normalized)

        logging.info(f"GP-DP covariance matrix shape: {cov_matrix.shape}")
        return sigma_allocations, query_locations

    def run(
            self,
            data: "Dataset",
            engine: Union["FactoredInference", "AdvancedSlicedInference"],
            workload: List[Tuple[str, ...]],
            records: Optional[int] = None,
    ) -> Tuple["Dataset", float]:
        """
        Run enhanced K-Way mechanism with selected noise type.

        Args:
            data (Dataset): Original dataset.
            workload (List[Tuple[str, ...]]): Query workload.
            engine: Inference engine.
            records (Optional[int]): Number of synthetic records.

        Returns:
            Tuple[Dataset, float]: Synthetic dataset and loss.
        """
        # Compute weights
        weights = self._get_weights(data, workload)

        logging.info(f"Using {self.noise_type} noise mechanism")
        logging.info(f"Weight strategy: {self.weight_strategy}")

        # Allocate noise based on mechanism type
        if self.noise_type == "renyi":
            sigma_allocations = self._allocate_noise_renyi(
                weights, workload, self.epsilon, self.delta
            )
            query_locations = None

        elif self.noise_type == "gp":
            sigma_allocations, query_locations = self._allocate_noise_gp(
                weights, workload, self.epsilon, self.delta, data
            )

        else:  # "gaussian"
            sigma_allocations = self._allocate_noise_gaussian(
                weights, workload, self.epsilon, self.delta
            )
            query_locations = None

        # Log statistics
        sigma_values = list(sigma_allocations.values())
        logging.info(f"Noise statistics:")
        logging.info(f"  Mean σ: {np.mean(sigma_values):.4f}")
        logging.info(f"  Min σ:  {np.min(sigma_values):.4f}")
        logging.info(f"  Max σ:  {np.max(sigma_values):.4f}")

        # Generate measurements
        total = data.records if self.bounded else None
        measurements = []

        for idx, cl in enumerate(workload):
            sigma = sigma_allocations[cl] * self.marginal_sensitivity

            Q = Identity(data.domain.size(cl))
            x = data.project(cl).datavector()

            # Generate noise based on mechanism
            if self.noise_type == "gp" and query_locations is not None:
                # For GP, generate correlated noise for this specific query
                # Note: Full GP would generate all noise jointly; this is a simplified version
                noise = self.prng.normal(loc=0, scale=sigma, size=x.size)
            else:
                noise = self.prng.normal(loc=0, scale=sigma, size=x.size)

            y = x + noise
            measurements.append((Q, y, sigma, cl))

        logging.info("Running inference engine...")
        est, loss = engine.estimate(measurements, total)

        logging.info("Generating synthetic data...")
        synth = est.synthetic_data(records)

        return synth, loss


# Backward compatibility
class KWayWeighted(KWayEnhanced):
    """Alias for backward compatibility."""

    def __init__(
            self,
            epsilon: float,
            delta: float = 0.00001,
            degree: int = 2,
            bounded: bool = True,
            prng: np.random = np.random,
            weight_strategy: str = "uniform",
            custom_weights: Optional[Dict[Tuple[str, ...], float]] = None,
    ):
        super().__init__(
            epsilon=epsilon,
            delta=delta,
            degree=degree,
            bounded=bounded,
            prng=prng,
            weight_strategy=weight_strategy,
            custom_weights=custom_weights,
            noise_type="gaussian"
        )


class KWay(KWayEnhanced):
    """Original K-Way for full backward compatibility."""

    def __init__(
            self,
            epsilon: float,
            delta: float = 0.00001,
            degree: int = 2,
            bounded: bool = True,
            prng: np.random = np.random,
    ):
        super().__init__(
            epsilon=epsilon,
            delta=delta,
            degree=degree,
            bounded=bounded,
            prng=prng,
            weight_strategy="uniform",
            noise_type="gaussian"
        )
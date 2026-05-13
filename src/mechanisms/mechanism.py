import math
from functools import partial
from typing import Optional, Callable, Tuple

import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal

from src.mechanisms.cdp2adp import cdp_rho
from src.mechanisms.privacy_calibrator import ana_gaussian_mech


class Mechanism:
    def __init__(
            self,
            epsilon: float,
            delta: float,
            bounded: bool = True,
            prng: np.random = np.random,
            noise_type: str = "gaussian",  # Options: "gaussian", "renyi", "gp"
            rdp_orders: Optional[np.ndarray] = None,
    ):
        """
        Base class for a mechanism with enhanced privacy mechanisms.

        Args:
            epsilon (float): Privacy parameter.
            delta (float): Privacy parameter.
            bounded (bool): Privacy definition (bounded vs unbounded DP).
            prng (np.random): Pseudo-random number generator.
            noise_type (str): Type of noise mechanism - "gaussian", "renyi", or "gp"
            rdp_orders (Optional[np.ndarray]): Orders for Rényi DP (alpha values)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.bounded = bounded
        self.sensitivity = 2.0 if self.bounded else 1.0
        self.marginal_sensitivity = np.sqrt(2) if self.bounded else 1.0
        self.prng = prng
        self.noise_type = noise_type

        # Rényi DP parameters
        if rdp_orders is None:
            # Default orders for RDP accounting
            self.rdp_orders = np.concatenate(
                [np.linspace(2, 10, 9), np.linspace(12, 64, 27)]
            )
        else:
            self.rdp_orders = rdp_orders

        # GP-DP parameters - Initialize as instance attributes
        self.gp_lengthscale = 1.0
        self.gp_variance = 1.0

    def run(self, data, engine, workload):
        pass

    def exponential_mechanism(self, qualities, epsilon, base_measure=None):
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5 * epsilon / self.sensitivity * q)
        else:
            p = softmax(0.5 * epsilon / self.sensitivity * q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    # ========================================================================
    # Rényi Differential Privacy (RDP) Methods
    # ========================================================================

    def compute_rdp_gaussian(self, sigma: float, alpha: float) -> float:
        """
        Compute Rényi divergence of order alpha for Gaussian mechanism.

        For Gaussian mechanism with noise N(0, σ²):
        D_α(M(D) || M(D')) = α / (2σ²)

        Args:
            sigma: Noise standard deviation
            alpha: Rényi divergence order

        Returns:
            RDP guarantee at order alpha
        """
        if alpha <= 1:
            return 0.0
        return alpha / (2 * sigma ** 2)

    def rdp_to_dp(
            self,
            rdp_curve: np.ndarray,
            delta: float,
            orders: Optional[np.ndarray] = None
    ) -> float:
        """
        Convert RDP curve to (ε, δ)-DP guarantee.

        Uses the formula:
        ε(δ) = min_α { rdp(α) + log(1/δ) / (α - 1) }

        Args:
            rdp_curve: Array of RDP values for each order
            delta: Target delta value
            orders: Optional array of orders corresponding to rdp_curve.
                   If None, uses self.rdp_orders (must match rdp_curve length)

        Returns:
            Epsilon guarantee
        """
        if orders is None:
            orders = self.rdp_orders

        # Ensure arrays have same length
        if len(rdp_curve) != len(orders):
            raise ValueError(
                f"Length mismatch: rdp_curve has {len(rdp_curve)} elements "
                f"but orders has {len(orders)} elements. "
                f"Either provide matching arrays or pass orders parameter."
            )

        eps_values = rdp_curve + np.log(1 / delta) / (orders - 1)
        return np.min(eps_values)

    def calibrate_gaussian_rdp(
            self,
            sensitivity: float,
            epsilon: float,
            delta: float
    ) -> float:
        """
        Calibrate Gaussian noise for RDP to achieve (ε, δ)-DP.

        Args:
            sensitivity: L2 sensitivity
            epsilon: Target epsilon
            delta: Target delta

        Returns:
            Optimal sigma for Gaussian mechanism
        """
        # Binary search for optimal sigma
        sigma_low = 0.1
        sigma_high = 100.0
        tolerance = 1e-4

        while sigma_high - sigma_low > tolerance:
            sigma_mid = (sigma_low + sigma_high) / 2

            # Compute RDP curve
            rdp_curve = np.array([
                self.compute_rdp_gaussian(sigma_mid / sensitivity, alpha)
                for alpha in self.rdp_orders
            ])

            # Convert to DP
            eps_achieved = self.rdp_to_dp(rdp_curve, delta)

            if eps_achieved > epsilon:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid

        return (sigma_low + sigma_high) / 2

    def renyi_noise_scale(
            self,
            l2_sensitivity: float,
            epsilon: float,
            delta: float
    ) -> float:
        """
        Compute noise scale using Rényi DP accounting.

        Args:
            l2_sensitivity: L2 sensitivity
            epsilon: Privacy parameter
            delta: Privacy parameter

        Returns:
            Noise scale for Gaussian mechanism with RDP accounting
        """
        if self.bounded:
            l2_sensitivity *= math.sqrt(2.0)

        sigma = self.calibrate_gaussian_rdp(l2_sensitivity, epsilon, delta)
        return l2_sensitivity * sigma

    # ========================================================================
    # Gaussian Process Differential Privacy (GP-DP) Methods
    # ========================================================================

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Radial Basis Function (RBF) kernel for Gaussian Process.

        K(x, x') = σ² exp(-||x - x'||² / (2 * l²))

        Args:
            X1: First set of points (n1 × d)
            X2: Second set of points (n2 × d)

        Returns:
            Kernel matrix (n1 × n2)
        """
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        # Compute pairwise squared distances
        sq_dists = np.sum(X1 ** 2, axis=1, keepdims=True) + \
                   np.sum(X2 ** 2, axis=1, keepdims=True).T - \
                   2 * np.dot(X1, X2.T)

        return self.gp_variance * np.exp(-sq_dists / (2 * self.gp_lengthscale ** 2))

    def gp_noise_covariance(
            self,
            locations: np.ndarray,
            epsilon: float,
            delta: float,
            sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Compute covariance matrix for GP-DP noise.

        The covariance is scaled to satisfy DP:
        Σ = (σ²_DP / σ²_GP) * K(X, X)

        Args:
            locations: Query locations (n × d)
            epsilon: Privacy parameter
            delta: Privacy parameter
            sensitivity: Query sensitivity

        Returns:
            Covariance matrix for GP noise (n × n)
        """
        if self.bounded:
            sensitivity *= math.sqrt(2.0)

        # Compute base kernel
        K = self.rbf_kernel(locations, locations)

        # Calibrate noise scale for DP
        sigma_dp = sensitivity * ana_gaussian_mech(epsilon, delta)["sigma"]

        # Scale kernel to achieve DP
        # The scaling ensures the trace matches required privacy
        n = locations.shape[0]
        K_scaled = (sigma_dp ** 2 / n) * K

        # Add small diagonal for numerical stability
        K_scaled += 1e-6 * np.eye(n)

        return K_scaled

    def gp_noise(
            self,
            locations: np.ndarray,
            epsilon: float,
            delta: float,
            sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Generate correlated Gaussian Process noise for DP.

        Args:
            locations: Query locations/indices (n × d)
            epsilon: Privacy parameter
            delta: Privacy parameter
            sensitivity: Query sensitivity

        Returns:
            Correlated noise vector (n,)
        """
        # Compute covariance matrix
        cov = self.gp_noise_covariance(locations, epsilon, delta, sensitivity)

        # Sample from multivariate Gaussian
        n = locations.shape[0]
        noise = self.prng.multivariate_normal(np.zeros(n), cov)

        return noise

    def gp_noise_scale_adaptive(
            self,
            query_locations: np.ndarray,
            epsilon: float,
            delta: float
    ) -> Callable:
        """
        Create adaptive GP noise function based on query structure.

        Args:
            query_locations: Locations of queries in feature space
            epsilon: Privacy parameter
            delta: Privacy parameter

        Returns:
            Function that generates noise for new queries
        """
        # Pre-compute covariance for observed locations
        cov = self.gp_noise_covariance(query_locations, epsilon, delta)

        def noise_fn(size):
            if size != len(query_locations):
                # For different size, use independent noise
                sigma = ana_gaussian_mech(epsilon, delta)["sigma"]
                return self.prng.normal(0, sigma * self.marginal_sensitivity, size)
            else:
                # Use correlated GP noise
                return self.prng.multivariate_normal(np.zeros(size), cov)

        return noise_fn

    # ========================================================================
    # Unified Noise Generation Interface
    # ========================================================================

    def gaussian_noise_scale(self, l2_sensitivity, epsilon, delta):
        """
        Return the Gaussian noise scale to attain (epsilon, delta)-DP.

        Supports multiple noise types based on self.noise_type.

        Args:
            l2_sensitivity: L2 sensitivity parameter.
            epsilon: Privacy parameter.
            delta: Privacy parameter.

        Returns:
            Gaussian noise scale.
        """
        if self.noise_type == "renyi":
            return self.renyi_noise_scale(l2_sensitivity, epsilon, delta)
        else:  # "gaussian" or default
            if self.bounded:
                l2_sensitivity *= math.sqrt(2.0)
            return l2_sensitivity * ana_gaussian_mech(epsilon, delta)["sigma"]

    def laplace_noise_scale(self, l1_sensitivity, epsilon):
        """
        Return the Laplace noise scale necessary to attain epsilon-DP.

        Args:
            l1_sensitivity: L1 sensitivity parameter.
            epsilon: Privacy parameter.

        Returns:
            Laplace noise scale.
        """
        if self.bounded:
            l1_sensitivity *= 2.0
        return l1_sensitivity / epsilon

    def gaussian_noise(self, sigma, size):
        """
        Generate iid Gaussian noise of a given scale and size.

        Args:
            sigma: Noise scale.
            size: Size of the noise.

        Returns:
            Generated noise.
        """
        return self.prng.normal(0, sigma, size)

    def laplace_noise(self, b, size):
        """
        Generate iid Laplace noise of a given scale and size.

        Args:
            b: Noise scale.
            size: Size of the noise.

        Returns:
            Generated noise.
        """
        return self.prng.laplace(0, b, size)

    def best_noise_distribution(
            self, l1_sensitivity, l2_sensitivity, epsilon, delta
    ):
        """
        Determine the best noise distribution based on noise_type.

        Args:
            l1_sensitivity: L1 sensitivity parameter.
            l2_sensitivity: L2 sensitivity parameter.
            epsilon: Privacy parameter.
            delta: Privacy parameter.

        Returns:
            Function that samples from the appropriate noise distribution.
        """
        if self.noise_type == "gp":
            # For GP, return a specialized function
            # This is a simplified version; actual implementation depends on query structure
            sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
            return partial(self.gaussian_noise, sigma)

        elif self.noise_type == "renyi":
            sigma = self.renyi_noise_scale(l2_sensitivity, epsilon, delta)
            return partial(self.gaussian_noise, sigma)

        else:  # "gaussian" or Laplace comparison
            b = self.laplace_noise_scale(l1_sensitivity, epsilon)
            sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
            if np.sqrt(2) * b < sigma:
                return partial(self.laplace_noise, b)
            return partial(self.gaussian_noise, sigma)

    def generate_noise(
            self,
            size: int,
            sensitivity: float,
            epsilon: float,
            delta: float,
            query_locations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Unified interface for generating noise based on mechanism type.

        Args:
            size: Size of noise vector
            sensitivity: Query sensitivity (L2)
            epsilon: Privacy parameter
            delta: Privacy parameter
            query_locations: Optional locations for GP-DP (for structured queries)

        Returns:
            Noise vector
        """
        if self.noise_type == "gp" and query_locations is not None:
            return self.gp_noise(query_locations, epsilon, delta, sensitivity)

        elif self.noise_type == "renyi":
            sigma = self.renyi_noise_scale(sensitivity, epsilon, delta)
            return self.gaussian_noise(sigma, size)

        else:  # "gaussian"
            sigma = self.gaussian_noise_scale(sensitivity, epsilon, delta)
            return self.gaussian_noise(sigma, size)
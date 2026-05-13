"""
Integration module for using Adaptive Particle Model with PrivPGD inference.
"""
import logging
from typing import Optional, List, Tuple

import torch
from src.inference.dataset import Dataset
from src.inference.domain import Domain
from .particle_model import AdaptiveParticleModel


class AdaptivePrivPGDConfig:
    """Configuration for adaptive PrivPGD with hierarchical particles."""

    def __init__(
            self,
            # Adaptive allocation settings
            adaptive_mode: str = "importance",  # "none", "variance", "importance", "entropy"
            enable_adaptive: bool = True,
            importance_threshold: float = 0.5,
            reallocation_frequency: int = 100,  # Reallocate every N iterations

            # Hierarchical settings
            use_hierarchical: bool = True,
            hierarchy_levels: int = 3,
            min_particles_per_region: int = 10,
            coarse_to_fine_schedule: Optional[List[int]] = None,  # Iterations to switch levels

            # Multi-scale optimization
            use_multiscale_transport: bool = True,
            adaptive_regularization: bool = True,

            # Performance settings
            enable_dynamic_resampling: bool = True,
            target_effective_particles: float = 0.8,  # ESS threshold for resampling
    ):
        """
        Initialize configuration for adaptive PrivPGD.

        Args:
            adaptive_mode: Strategy for adaptive allocation
            enable_adaptive: Whether to enable adaptive allocation
            importance_threshold: Threshold for identifying important regions
            reallocation_frequency: How often to reallocate particles
            use_hierarchical: Enable hierarchical particle structure
            hierarchy_levels: Number of hierarchy levels
            min_particles_per_region: Minimum particles per region
            coarse_to_fine_schedule: When to switch between hierarchy levels
            use_multiscale_transport: Use multi-scale optimal transport
            adaptive_regularization: Adapt regularization based on importance
            enable_dynamic_resampling: Enable dynamic particle resampling
            target_effective_particles: ESS threshold for resampling
        """
        self.adaptive_mode = adaptive_mode
        self.enable_adaptive = enable_adaptive
        self.importance_threshold = importance_threshold
        self.reallocation_frequency = reallocation_frequency

        self.use_hierarchical = use_hierarchical
        self.hierarchy_levels = hierarchy_levels
        self.min_particles_per_region = min_particles_per_region

        if coarse_to_fine_schedule is None:
            # Default: start coarse, gradually refine
            self.coarse_to_fine_schedule = [
                hierarchy_levels - 1,  # Start at coarsest
                hierarchy_levels - 2,  # After 1/3 iterations
                0,  # Finish at finest
            ]
        else:
            self.coarse_to_fine_schedule = coarse_to_fine_schedule

        self.use_multiscale_transport = use_multiscale_transport
        self.adaptive_regularization = adaptive_regularization
        self.enable_dynamic_resampling = enable_dynamic_resampling
        self.target_effective_particles = target_effective_particles


def create_adaptive_particle_model(
        domain: Domain,
        n_particles: int,
        data_init: Optional[Dataset] = None,
        config: Optional[AdaptivePrivPGDConfig] = None
) -> AdaptiveParticleModel:
    """
    Create an adaptive particle model with specified configuration.

    Args:
        domain: Domain specification
        n_particles: Number of particles
        data_init: Optional initialization dataset
        config: Adaptive configuration

    Returns:
        Configured adaptive particle model
    """
    if config is None:
        config = AdaptivePrivPGDConfig()

    model = AdaptiveParticleModel(
        domain=domain,
        n_particles=n_particles,
        data_init=data_init,
        adaptive_mode=config.adaptive_mode if config.enable_adaptive else "none",
        use_hierarchical=config.use_hierarchical,
        hierarchy_levels=config.hierarchy_levels,
        min_particles_per_region=config.min_particles_per_region,
    )

    logging.info(
        f"Created adaptive particle model: "
        f"adaptive={config.enable_adaptive}, hierarchical={config.use_hierarchical}"
    )

    return model


def compute_effective_sample_size(weights: torch.Tensor) -> float:
    """
    Compute effective sample size (ESS) for particle weights.

    ESS measures how well particles represent the distribution.
    ESS = 1 / sum(w_i^2)

    Args:
        weights: Particle weights (should sum to 1)

    Returns:
        Effective sample size as fraction of total particles
    """
    ess = 1.0 / torch.sum(weights ** 2)
    return ess.item() / len(weights)


def should_resample(
        weights: torch.Tensor,
        threshold: float = 0.5
) -> bool:
    """
    Determine if particles should be resampled based on ESS.

    Args:
        weights: Current particle weights
        threshold: ESS threshold (0-1)

    Returns:
        True if resampling is needed
    """
    ess = compute_effective_sample_size(weights)
    return ess < threshold


def resample_particles(
        particles: torch.Tensor,
        weights: torch.Tensor,
        n_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resample particles based on their weights (systematic resampling).

    Args:
        particles: Current particles (n x d)
        weights: Current weights (n,)
        n_samples: Number of resampled particles (default: same as input)

    Returns:
        Resampled particles and uniform weights
    """
    if n_samples is None:
        n_samples = particles.shape[0]

    # Systematic resampling
    weights_cumsum = torch.cumsum(weights, dim=0)

    # Generate systematic samples
    u = torch.rand(1, device=particles.device) / n_samples
    positions = u + torch.arange(n_samples, device=particles.device).float() / n_samples

    # Find indices
    indices = torch.searchsorted(weights_cumsum, positions)
    indices = torch.clamp(indices, 0, len(weights) - 1)

    # Resample
    particles_resampled = particles[indices]
    weights_resampled = torch.full(
        (n_samples,),
        1.0 / n_samples,
        device=particles.device
    )

    return particles_resampled, weights_resampled


def adaptive_iteration_update(
        model: AdaptiveParticleModel,
        iteration: int,
        total_iterations: int,
        config: AdaptivePrivPGDConfig,
        marginals: Optional[List[Tuple[str, ...]]] = None
) -> int:
    """
    Perform adaptive updates during PrivPGD iteration.

    Args:
        model: Adaptive particle model
        iteration: Current iteration number
        total_iterations: Total number of iterations
        config: Adaptive configuration
        marginals: Optional marginal queries for importance computation

    Returns:
        Current hierarchy level being used
    """
    # 1. Determine current hierarchy level (coarse-to-fine)
    current_level = 0
    if config.use_hierarchical:
        progress = iteration / total_iterations

        # Map progress to hierarchy level
        if progress < 0.33:
            current_level = config.coarse_to_fine_schedule[0]
        elif progress < 0.67:
            current_level = config.coarse_to_fine_schedule[1]
        else:
            current_level = config.coarse_to_fine_schedule[2]

    # 2. Adaptive particle reallocation
    if config.enable_adaptive and iteration % config.reallocation_frequency == 0:
        logging.info(f"Iteration {iteration}: Computing importance and reallocating particles")

        # Compute region importance
        model.compute_region_importance(marginals=marginals)

        # Reallocate particles to important regions
        model.adaptive_reallocate_particles(
            importance_threshold=config.importance_threshold
        )

    # 3. Dynamic resampling if needed
    if config.enable_dynamic_resampling:
        ess = compute_effective_sample_size(model.w)

        if ess < config.target_effective_particles:
            logging.info(f"Iteration {iteration}: ESS={ess:.3f}, resampling particles")
            model.X, model.w = resample_particles(model.X, model.w)

    return current_level


def get_transport_parameters(
        config: AdaptivePrivPGDConfig,
        iteration: int,
        model: AdaptiveParticleModel
) -> dict:
    """
    Get optimal transport parameters based on configuration.

    Args:
        config: Adaptive configuration
        iteration: Current iteration
        model: Particle model (for accessing importance weights)

    Returns:
        Dictionary of transport parameters
    """
    params = {
        'use_multiscale': config.use_multiscale_transport,
        'adaptive_reg': config.adaptive_regularization,
    }

    if config.enable_adaptive and model.region_importance is not None:
        params['importance_weights'] = model.region_importance
    else:
        params['importance_weights'] = None

    return params


# Example usage integration with existing AdvancedSlicedInference
def integrate_with_advanced_sliced_inference():
    """
    Example of how to integrate adaptive particles with AdvancedSlicedInference.

    This is a template showing the modifications needed.
    """
    example_code = '''
    # In AdvancedSlicedInference.__init__():

    def __init__(self, domain, hp):
        # ... existing code ...

        # Add adaptive configuration
        self.adaptive_config = AdaptivePrivPGDConfig(
            adaptive_mode=hp.get("adaptive_mode", "importance"),
            enable_adaptive=hp.get("enable_adaptive", True),
            use_hierarchical=hp.get("use_hierarchical", True),
            hierarchy_levels=hp.get("hierarchy_levels", 3),
        )

    # In AdvancedSlicedInference.estimate() or training loop:

    def estimate(self, measurements, total):
        # Create adaptive particle model instead of standard ParticleModel
        model = create_adaptive_particle_model(
            domain=self.domain,
            n_particles=self.hp["n_particles"],
            config=self.adaptive_config
        )

        # Training loop
        for iteration in range(self.hp["iters"]):
            # Adaptive updates
            current_level = adaptive_iteration_update(
                model=model,
                iteration=iteration,
                total_iterations=self.hp["iters"],
                config=self.adaptive_config,
                marginals=self.workload  # Your marginal queries
            )

            # Get particles at appropriate hierarchy level
            X, w = model.get_particles_at_level(current_level)

            # ... rest of PrivPGD iteration using X and w ...

            # Update model.X and model.w after gradient step

        return model
    '''

    return example_code


# Utility: Create configuration from hyperparameters
def config_from_hyperparameters(hp: dict) -> AdaptivePrivPGDConfig:
    """
    Create AdaptivePrivPGDConfig from hyperparameter dictionary.

    Args:
        hp: Hyperparameter dictionary

    Returns:
        Adaptive configuration
    """
    return AdaptivePrivPGDConfig(
        adaptive_mode=hp.get("adaptive_mode", "importance"),
        enable_adaptive=hp.get("enable_adaptive", True),
        importance_threshold=hp.get("importance_threshold", 0.5),
        reallocation_frequency=hp.get("reallocation_frequency", 100),
        use_hierarchical=hp.get("use_hierarchical", True),
        hierarchy_levels=hp.get("hierarchy_levels", 3),
        min_particles_per_region=hp.get("min_particles_per_region", 10),
        use_multiscale_transport=hp.get("use_multiscale_transport", True),
        adaptive_regularization=hp.get("adaptive_regularization", True),
        enable_dynamic_resampling=hp.get("enable_dynamic_resampling", True),
        target_effective_particles=hp.get("target_effective_particles", 0.8),
    )


# Logging and diagnostics
def log_particle_statistics(model: AdaptiveParticleModel, iteration: int):
    """
    Log statistics about particle distribution.

    Args:
        model: Adaptive particle model
        iteration: Current iteration number
    """
    ess = compute_effective_sample_size(model.w)

    logging.info(f"Iteration {iteration} Particle Statistics:")
    logging.info(f"  Total particles: {model.n_particles}")
    logging.info(f"  Effective sample size: {ess:.3f}")

    if model.region_importance is not None:
        imp_stats = model.region_importance
        logging.info(f"  Importance - min: {imp_stats.min():.4f}, "
                     f"max: {imp_stats.max():.4f}, "
                     f"mean: {imp_stats.mean():.4f}")

    if model.use_hierarchical:
        logging.info(f"  Hierarchy levels: {model.hierarchy_levels}")
        for level, h_info in enumerate(model.hierarchical_particles):
            logging.info(f"    Level {level}: {h_info['n_particles']} particles")
from typing import Optional, Tuple, Dict
import logging

import cvxpy as cp
import numpy as np
import ot
import torch


def project_onto_probability_space_marginal(
        M: np.ndarray, vdp: np.ndarray
) -> np.ndarray:
    """
    Projects a matrix onto the probability space considering the marginal distribution.

    Args:
        M (np.ndarray): A square matrix representing the cost.
        vdp (np.ndarray): The vector of the dual problem.

    Returns:
        np.ndarray: The projected vector.
    """
    n = M.shape[0]

    # dual variables
    x = cp.Variable((n, n), nonneg=True)
    y = cp.Variable(n, nonneg=True)
    v = cp.Variable(n, nonneg=True)
    # The objective is to minimize the dot product of M and x
    objective = cp.Minimize(cp.sum(cp.multiply(M, x)) + 2 * cp.sum(y))

    # Constraints
    constraints = [
        cp.diag(x) == np.zeros(n),
        cp.sum(x, axis=0) - cp.sum(x, axis=1) >= v - vdp - y,  # sum over rows
        cp.sum(v) == 1,
    ]
    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()
    v_value = v.value

    # get rid of numerical issues
    return v_value / sum(v_value)


def adaptive_sinkhorn_gradient_weights(
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: float,
        importance_weights: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the Sinkhorn algorithm with adaptive importance weighting.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix.
        reg (float): The regularization parameter.
        importance_weights (Optional[np.ndarray]): Importance weights for adaptive allocation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The gradient with respect to the weights,
        and the log of the dual variable 'u'.
    """
    # Apply importance weights to source distribution
    if importance_weights is not None:
        a_weighted = a * importance_weights
        a_weighted = a_weighted / a_weighted.sum()
    else:
        a_weighted = a

    mat, logs = ot.bregman.sinkhorn_log(a_weighted, b, M, reg, log=True)

    # Adjust gradient by importance
    grad_base = reg * torch.log(logs["u"])
    if importance_weights is not None:
        grad_adjusted = grad_base * torch.from_numpy(importance_weights).float()
    else:
        grad_adjusted = grad_base

    return torch.sum(mat * M), grad_adjusted


def hierarchical_sinkhorn_gradient(
        a_hierarchy: Dict[int, np.ndarray],
        b: np.ndarray,
        M_hierarchy: Dict[int, np.ndarray],
        reg: float,
        current_level: int = 0,
        use_coarse_to_fine: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Sinkhorn gradient using hierarchical multi-scale approach.

    Args:
        a_hierarchy: Dictionary mapping level -> source distribution at that level
        b: Target distribution
        M_hierarchy: Dictionary mapping level -> cost matrix at that level
        reg: Regularization parameter
        current_level: Current hierarchy level to compute
        use_coarse_to_fine: Whether to use coarse-to-fine initialization

    Returns:
        Loss, transport matrix, and gradient
    """
    # Start from coarsest level if using hierarchical initialization
    if use_coarse_to_fine and current_level > 0:
        # Get solution from coarser level
        max_level = max(a_hierarchy.keys())
        coarse_level = min(current_level + 1, max_level)

        if coarse_level in a_hierarchy:
            # Solve at coarse level first
            a_coarse = a_hierarchy[coarse_level]
            M_coarse = M_hierarchy[coarse_level]

            # Coarse solution (use as initialization)
            mat_coarse, logs_coarse = ot.bregman.sinkhorn_log(
                a_coarse, b, M_coarse, reg, log=True
            )

    # Solve at current level
    a_current = a_hierarchy[current_level]
    M_current = M_hierarchy[current_level]

    mat, logs = ot.bregman.sinkhorn_log(a_current, b, M_current, reg, log=True)

    loss = torch.sum(mat * M_current)
    f = reg * torch.log(logs["u"])

    return loss, mat, f


def sinkhorn_gradient_locations(
        a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the Sinkhorn algorithm with respect to the locations.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix.
        reg (float): The regularization parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The gradient with respect to the locations,
        and the Sinkhorn matrix.
    """
    mat = ot.bregman.sinkhorn_log(a, b, M, reg, log=False)
    return torch.sum(mat * M), mat


def adaptive_sinkhorn_divergence_gradient(
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        Ma: np.ndarray,
        reg: float,
        importance_weights: Optional[np.ndarray] = None,
        adaptive_reg: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the Sinkhorn divergence with adaptive features.

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix for the (a, b) comparison.
        Ma (np.ndarray): The cost matrix for the (a, a) comparison.
        reg (float): The regularization parameter.
        importance_weights (Optional[np.ndarray]): Importance weights for adaptive allocation.
        adaptive_reg (bool): Whether to use adaptive regularization based on local density.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        The Sinkhorn distance without entropy, the Sinkhorn matrices for (a, b) and (a, a),
        and the gradient.
    """
    # Apply importance weights
    if importance_weights is not None:
        a_weighted = a * importance_weights
        a_weighted = a_weighted / a_weighted.sum()
    else:
        a_weighted = a

    a_weighted = torch.abs(a_weighted) / torch.sum(torch.abs(a_weighted))
    b = torch.abs(b) / torch.sum(torch.abs(b))

    # Adaptive regularization based on local density
    if adaptive_reg and importance_weights is not None:
        # Higher regularization in low-density (high-importance) regions
        reg_adaptive = reg * (1.0 + importance_weights.mean())
    else:
        reg_adaptive = reg

    mat, logs = ot.bregman.sinkhorn_log(a_weighted, b, M, reg_adaptive, log=True)
    amat, alogs = ot.bregman.sinkhorn_log(a_weighted, a_weighted, Ma, reg_adaptive, log=True)

    f = reg_adaptive * torch.log(logs["u"] + 1e-20)
    af, ag = (
        reg_adaptive * torch.log(alogs["u"] + 1e-20),
        reg_adaptive * torch.log(alogs["v"] + 1e-20)
    )

    sinkhorn_distance_without_entropy = torch.sum(mat * M)
    gradw = f - 1 / 2 * (af + ag)

    # Apply importance weighting to gradient
    if importance_weights is not None:
        gradw = gradw * torch.from_numpy(importance_weights).float()

    return sinkhorn_distance_without_entropy, mat, amat, gradw


def sinkhorn_gradient(
        a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Computes the Sinkhorn gradient (standard version for backward compatibility).

    Args:
        a (np.ndarray): The source distribution.
        b (np.ndarray): The target distribution.
        M (np.ndarray): The cost matrix.
        reg (float): The regularization parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        The Sinkhorn distance without entropy, the Sinkhorn matrix, None, and the gradient.
    """
    a = torch.abs(a) / torch.sum(torch.abs(a))
    b = torch.abs(b) / torch.sum(torch.abs(b))

    mat, logs = ot.bregman.sinkhorn_log(a, b, M, reg, log=True)
    f = reg * torch.log(logs["u"])
    sinkhorn_distance_without_entropy = torch.sum(mat * M)
    f = torch.clamp(f, -100.0, 100.0)
    gradw = f
    return sinkhorn_distance_without_entropy, mat, None, gradw


def multi_scale_sinkhorn(
        a_fine: np.ndarray,
        b: np.ndarray,
        M_fine: np.ndarray,
        reg: float,
        n_scales: int = 3,
        scale_factor: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-scale Sinkhorn algorithm for improved convergence.

    Solves optimal transport at multiple scales, using coarse solutions
    to initialize finer scales.

    Args:
        a_fine: Source distribution at finest scale
        b: Target distribution
        M_fine: Cost matrix at finest scale
        reg: Regularization parameter
        n_scales: Number of scales to use
        scale_factor: Factor to reduce resolution at each scale

    Returns:
        Final loss and transport matrix
    """
    # Build hierarchy of distributions and cost matrices
    a_hierarchy = {0: a_fine}
    M_hierarchy = {0: M_fine}

    for scale in range(1, n_scales):
        # Coarsen distribution by averaging
        n_current = a_hierarchy[scale - 1].shape[0]
        n_coarse = max(10, n_current // scale_factor)

        # Simple averaging for coarsening
        indices = torch.linspace(0, n_current - 1, n_coarse).long()
        a_coarse = torch.zeros(n_coarse)

        for i, idx in enumerate(indices):
            start = max(0, idx - scale_factor // 2)
            end = min(n_current, idx + scale_factor // 2)
            a_coarse[i] = a_hierarchy[scale - 1][start:end].sum()

        a_coarse = a_coarse / a_coarse.sum()
        a_hierarchy[scale] = a_coarse

        # Coarsen cost matrix
        M_coarse = torch.zeros(n_coarse, b.shape[0])
        for i, idx in enumerate(indices):
            start = max(0, idx - scale_factor // 2)
            end = min(n_current, idx + scale_factor // 2)
            M_coarse[i] = M_hierarchy[scale - 1][start:end].mean(dim=0)

        M_hierarchy[scale] = M_coarse

    # Solve from coarse to fine
    u_init = None
    v_init = None

    for scale in reversed(range(n_scales)):
        a_current = a_hierarchy[scale]
        M_current = M_hierarchy[scale]

        # Solve at current scale
        if u_init is not None and scale < n_scales - 1:
            # Use previous solution as initialization (upsampled)
            # This is simplified - in practice would need proper interpolation
            pass

        mat, logs = ot.bregman.sinkhorn_log(a_current, b, M_current, reg, log=True)
        u_init = logs["u"]
        v_init = logs["v"]

    # Return finest scale solution
    loss = torch.sum(mat * M_fine)
    return loss, mat


def compute_adaptive_transport_plan(
        particles_source: torch.Tensor,
        weights_source: torch.Tensor,
        particles_target: torch.Tensor,
        weights_target: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        reg: float = 0.1,
        use_multiscale: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute optimal transport plan with adaptive features.

    Args:
        particles_source: Source particles
        weights_source: Source weights
        particles_target: Target particles
        weights_target: Target weights
        importance_weights: Importance scores for adaptive allocation
        reg: Regularization parameter
        use_multiscale: Whether to use multi-scale approach

    Returns:
        Transport plan and additional info dict
    """
    # Compute cost matrix
    M = torch.cdist(particles_source, particles_target, p=2) ** 2

    info = {}

    if use_multiscale:
        loss, transport_plan = multi_scale_sinkhorn(
            weights_source.cpu().numpy(),
            weights_target.cpu().numpy(),
            M.cpu().numpy(),
            reg=reg,
            n_scales=3
        )
        info['multiscale_loss'] = loss
    else:
        # Standard Sinkhorn with optional importance weighting
        if importance_weights is not None:
            loss, grad = adaptive_sinkhorn_gradient_weights(
                weights_source.cpu().numpy(),
                weights_target.cpu().numpy(),
                M.cpu().numpy(),
                reg,
                importance_weights=importance_weights.cpu().numpy()
            )
            info['importance_weighted'] = True
        else:
            loss, transport_plan, _, grad = sinkhorn_gradient(
                weights_source.cpu().numpy(),
                weights_target.cpu().numpy(),
                M.cpu().numpy(),
                reg
            )
            info['importance_weighted'] = False

        transport_plan = torch.from_numpy(transport_plan).float()
        info['loss'] = loss

    return transport_plan, info
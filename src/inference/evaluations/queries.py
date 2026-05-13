import numpy as np


def counting_query_error(
        real_df,
        synth_df,
        n_queries=100,
        sparsity=3,
        min_frac=0.05,
        max_frac=0.95,
        seed=0,
):
    rng = np.random.default_rng(seed)
    Xr = real_df.values
    Xs = synth_df.values
    n, d = Xr.shape

    errors = []

    # ✅ 对于离散化数据，自动调整参数
    if all(len(np.unique(Xr[:, j])) <= 20 for j in range(d)):
        sparsity = min(sparsity, 2)  # 减少维度
        min_frac = 0.01  # 放宽下限
        max_frac = 0.99  # 放宽上限

    for _ in range(n_queries):
        dims = rng.choice(d, size=sparsity, replace=False)

        # rejection sampling for interval
        for _ in range(100):
            mask = np.ones(n, dtype=bool)
            bounds = {}

            for j in dims:
                col = Xr[:, j]
                unique_vals = np.unique(col)

                # ✅ 如果是离散值，使用离散采样
                if len(unique_vals) <= 10:
                    n_select = rng.integers(1, len(unique_vals) + 1)
                    selected = rng.choice(unique_vals, size=n_select, replace=False)
                    mask &= np.isin(col, selected)
                    bounds[j] = selected
                else:
                    lo = rng.uniform(col.min(), col.max())
                    hi = rng.uniform(lo, col.max())
                    bounds[j] = (lo, hi)
                    mask &= (col >= lo) & (col <= hi)

            frac = mask.mean()
            if min_frac <= frac <= max_frac:
                break
        else:
            continue

        real_count = mask.mean()

        synth_mask = np.ones(len(Xs), dtype=bool)
        for j, bound in bounds.items():
            if isinstance(bound, np.ndarray):
                synth_mask &= np.isin(Xs[:, j], bound)
            else:
                lo, hi = bound
                synth_mask &= (Xs[:, j] >= lo) & (Xs[:, j] <= hi)

        synth_count = synth_mask.mean()
        errors.append(abs(real_count - synth_count))

    if len(errors) == 0:
        return {"counting_query_error": 0.0}

    return {"counting_query_error": float(np.mean(errors))}


def threshold_query_error(
        real_df,
        synth_df,
        n_queries=100,
        sparsity=3,
        seed=0,
):
    """
    Compute error on 3-sparse linear thresholding queries.

    From paper:
    thresh_j(D) = (1/n) * Σ_i 1_{<x_i, θ> + b_j > 0}

    Where:
    - θ is a random 3-sparse direction with values in {-1, +1}
    - b_j ~ Uniform[min_{x∈D}<x,θ>, max_{x∈D}<x,θ>]

    Args:
        real_df: Original dataset
        synth_df: Synthetic dataset
        n_queries: Number of random threshold queries to generate
        sparsity: Number of non-zero dimensions (paper uses 3)
        seed: Random seed for reproducibility

    Returns:
        dict: {"threshold_query_error": mean absolute error across queries}
    """
    rng = np.random.default_rng(seed)

    # Convert to numpy arrays
    Xr = real_df.values if hasattr(real_df, 'values') else np.array(real_df)
    Xs = synth_df.values if hasattr(synth_df, 'values') else np.array(synth_df)

    n_real, d = Xr.shape
    n_synth = Xs.shape[0]

    # Convert to float for numerical stability
    Xr = Xr.astype(float)
    Xs = Xs.astype(float)

    errors = []

    for query_idx in range(n_queries):
        # Sample sparsity-many dimensions uniformly at random
        dims = rng.choice(d, size=sparsity, replace=False)

        # Create sparse direction θ with values in {-1, +1}
        theta = np.zeros(d, dtype=float)
        theta[dims] = rng.choice([-1.0, 1.0], size=sparsity)

        # Project real data onto θ
        proj_real = Xr @ theta  # Shape: (n_real,)

        # Sample threshold b uniformly from [min_proj, max_proj] on REAL data
        min_proj = proj_real.min()
        max_proj = proj_real.max()

        # Handle edge case where all projections are identical
        if max_proj - min_proj < 1e-10:
            b = min_proj
        else:
            b = rng.uniform(min_proj, max_proj)

        # Compute threshold query on real data
        # thresh(D) = (1/n) * Σ 1_{<x_i, θ> + b > 0}
        real_count = np.sum(proj_real + b > 0)
        real_val = real_count / n_real

        # Compute threshold query on synthetic data
        proj_synth = Xs @ theta  # Shape: (n_synth,)
        synth_count = np.sum(proj_synth + b > 0)
        synth_val = synth_count / n_synth

        # Absolute error for this query
        error = abs(real_val - synth_val)
        errors.append(error)

    # Return mean absolute error across all queries
    mean_error = np.mean(errors)

    return {"threshold_query_error": float(mean_error)}
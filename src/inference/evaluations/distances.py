import itertools
import numpy as np

import numpy as np
import itertools

import numpy as np
import itertools


def sw1_distance(real_df, synth_df, num_projections=50, seed=0):
    """
    Compute average SW1 distance over all 2-way marginals with proper categorical embedding.

    Formula from paper:
    (d choose 2)^(-1) * Σ_{S⊂[d], |S|=2} SW1(μ_S[D], μ_S[D_DP])

    Key: Data must be embedded from categorical space to R^k before computing SW1.

    Args:
        real_df: Original dataset (categorical/discrete data)
        synth_df: Synthetic DP dataset
        num_projections: Number of random projections (paper likely uses more)
        seed: Random seed

    Returns:
        dict: {"sw1_distance": average SW1 over all 2-way marginals}
    """
    rng = np.random.default_rng(seed)

    # Convert to numpy arrays
    Xr = real_df.values if hasattr(real_df, 'values') else np.array(real_df)
    Xs = synth_df.values if hasattr(synth_df, 'values') else np.array(synth_df)

    n_real, d = Xr.shape
    n_synth = Xs.shape[0]

    # Step 1: Embed categorical data into continuous space
    # For each attribute, use one-hot encoding or ordinal embedding
    Xr_embedded = _embed_categorical(Xr)
    Xs_embedded = _embed_categorical(Xs)

    marginal_sw1_values = []

    # Iterate over all 2-way marginals
    for i, j in itertools.combinations(range(d), 2):
        # Extract embedded 2-way marginal
        # After embedding, each dimension might be multi-dimensional
        Xr_ij = Xr_embedded[:, [i, j]]
        Xs_ij = Xs_embedded[:, [i, j]]

        # Compute SW1 for this 2-way marginal
        sw1_marginal = _sliced_wasserstein_1(Xr_ij, Xs_ij, num_projections, rng)
        marginal_sw1_values.append(sw1_marginal)

    # Average over all 2-way marginals
    avg_sw1 = np.mean(marginal_sw1_values)

    return {"sw1_distance": float(avg_sw1)}


def _embed_categorical(X):
    """
    Embed categorical data into R^d.
    For discrete/categorical data, keep as ordinal but normalize to [0,1].

    Args:
        X: numpy array of shape (n, d) with integer categorical values

    Returns:
        X_embedded: numpy array of shape (n, d) normalized to [0,1]
    """
    X_embedded = X.astype(float).copy()

    # Normalize each dimension to [0, 1]
    for col in range(X.shape[1]):
        min_val = X_embedded[:, col].min()
        max_val = X_embedded[:, col].max()

        if max_val > min_val:
            X_embedded[:, col] = (X_embedded[:, col] - min_val) / (max_val - min_val)
        else:
            X_embedded[:, col] = 0.0

    return X_embedded


def _sliced_wasserstein_1(X1, X2, num_projections, rng):
    """
    Compute Sliced Wasserstein-1 distance between two point clouds.

    SW1(μ, ν) = ∫_{S^(d-1)} W1(P_θ#μ, P_θ#ν) dθ

    Args:
        X1: numpy array of shape (n1, k)
        X2: numpy array of shape (n2, k)
        num_projections: number of random projections
        rng: numpy random generator

    Returns:
        float: SW1 distance
    """
    k = X1.shape[1]

    w1_distances = []

    for _ in range(num_projections):
        # Sample random unit vector θ on unit sphere S^(k-1)
        theta = rng.normal(size=k)
        theta = theta / np.linalg.norm(theta)

        # Project data onto θ
        proj1 = X1 @ theta
        proj2 = X2 @ theta

        # Compute W1 distance between 1D distributions
        # W1(μ, ν) = ∫|F_μ^(-1)(u) - F_ν^(-1)(u)| du
        w1 = _wasserstein_1d(proj1, proj2)
        w1_distances.append(w1)

    # Average over projections
    return np.mean(w1_distances)


def _wasserstein_1d(x1, x2):
    """
    Compute exact 1-Wasserstein distance between two 1D empirical distributions.

    For empirical measures with equal weights:
    W1(μ, ν) = (1/n) * Σ|x_i - y_i| where x_i, y_i are sorted samples

    Args:
        x1: 1D array of samples from first distribution
        x2: 1D array of samples from second distribution

    Returns:
        float: W1 distance
    """
    # Sort both arrays
    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2)

    n1 = len(x1_sorted)
    n2 = len(x2_sorted)

    # For equal-sized samples (most common case)
    if n1 == n2:
        return np.mean(np.abs(x1_sorted - x2_sorted))

    # For unequal sizes, use CDF-based approach
    # This is more accurate than quantile approximation
    all_points = np.concatenate([x1_sorted, x2_sorted])
    all_points = np.unique(all_points)

    cdf1 = np.searchsorted(x1_sorted, all_points, side='right') / n1
    cdf2 = np.searchsorted(x2_sorted, all_points, side='right') / n2

    # W1 = ∫|F1(x) - F2(x)| dx (trapezoidal approximation)
    w1 = np.trapz(np.abs(cdf1 - cdf2), all_points)

    return w1


def sw1_distance_simple(real_df, synth_df, num_projections=500, seed=0):
    """
    Simplified version assuming data is already properly scaled.
    """
    rng = np.random.default_rng(seed)

    Xr = real_df.values if hasattr(real_df, 'values') else np.array(real_df)
    Xs = synth_df.values if hasattr(synth_df, 'values') else np.array(synth_df)

    # Normalize to [0, 1]
    Xr_norm = np.zeros_like(Xr, dtype=float)
    Xs_norm = np.zeros_like(Xs, dtype=float)

    for col in range(Xr.shape[1]):
        min_val = min(Xr[:, col].min(), Xs[:, col].min())
        max_val = max(Xr[:, col].max(), Xs[:, col].max())

        if max_val > min_val:
            Xr_norm[:, col] = (Xr[:, col] - min_val) / (max_val - min_val)
            Xs_norm[:, col] = (Xs[:, col] - min_val) / (max_val - min_val)

    d = Xr.shape[1]
    marginal_sw1_values = []

    for i, j in itertools.combinations(range(d), 2):
        Xr_ij = Xr_norm[:, [i, j]]
        Xs_ij = Xs_norm[:, [i, j]]

        w1_distances = []

        for _ in range(num_projections):
            theta = rng.normal(size=2)
            theta = theta / np.linalg.norm(theta)

            proj_r = Xr_ij @ theta
            proj_s = Xs_ij @ theta

            # Sort and compute exact W1
            proj_r_sorted = np.sort(proj_r)
            proj_s_sorted = np.sort(proj_s)

            if len(proj_r) == len(proj_s):
                w1 = np.mean(np.abs(proj_r_sorted - proj_s_sorted))
            else:
                # Use CDF-based calculation
                all_pts = np.unique(np.concatenate([proj_r_sorted, proj_s_sorted]))
                cdf_r = np.searchsorted(proj_r_sorted, all_pts, side='right') / len(proj_r)
                cdf_s = np.searchsorted(proj_s_sorted, all_pts, side='right') / len(proj_s)
                w1 = np.trapz(np.abs(cdf_r - cdf_s), all_pts)

            w1_distances.append(w1)

        marginal_sw1_values.append(np.mean(w1_distances))

    return {"sw1_distance": float(np.mean(marginal_sw1_values))}


import json


def tv_distance_final(real_df, synth_df, domain_path=None, workload=None):
    """
    Final TV distance with all fixes and robust domain parsing.
    """
    Xr = real_df.values.astype(int)
    Xs = synth_df.values.astype(int)

    n_real, d = Xr.shape
    n_synth = Xs.shape[0]

    # Determine cardinalities
    if domain_path is not None:
        # Use domain specification (RECOMMENDED for paper comparison)
        import json
        with open(domain_path, 'r') as f:
            domain = json.load(f)

        print(f"Domain type: {type(domain)}")
        print(f"Domain content (first 100 chars): {str(domain)[:100]}")

        # Handle different domain formats
        if isinstance(domain, dict):
            # Check if values are integers (cardinalities) or lists (actual values)
            first_key = list(domain.keys())[0]
            first_value = domain[first_key]

            if isinstance(first_value, int):
                # Format: {"col1": 10, "col2": 5, ...} - cardinalities directly
                cardinalities = [domain[col] for col in sorted(domain.keys())]
            elif isinstance(first_value, list):
                # Format: {"col1": [0,1,2,...], "col2": [0,1,2,...]} - value lists
                cardinalities = [len(domain[col]) for col in sorted(domain.keys())]
            else:
                raise ValueError(f"Unknown domain value format: {type(first_value)}")

        elif isinstance(domain, list):
            # Format: [10, 5, 8, ...] - list of cardinalities
            if all(isinstance(x, int) for x in domain):
                cardinalities = domain
            # Format: [[0,1,2,...], [0,1,2,...]] - list of value lists
            elif all(isinstance(x, list) for x in domain):
                cardinalities = [len(x) for x in domain]
            else:
                raise ValueError(f"Unknown domain list format")
        else:
            raise ValueError(f"Unknown domain format: {type(domain)}")

    elif workload is not None:
        # Extract from workload (if format is correct)
        cardinalities = {}
        for item in workload:
            try:
                if isinstance(item, tuple) and len(item) >= 2:
                    cols, shape = item[0], item[1]
                    if len(cols) == 2:
                        cardinalities[cols[0]] = shape[0]
                        cardinalities[cols[1]] = shape[1]
            except Exception as e:
                continue

        # Fill in missing cardinalities from data
        cardinalities = [cardinalities.get(i, max(Xr[:, i].max(), Xs[:, i].max()) + 1)
                         for i in range(d)]
    else:
        # Auto-detect from data
        cardinalities = [max(Xr[:, i].max(), Xs[:, i].max()) + 1 for i in range(d)]

    print(f"Using cardinalities: {cardinalities}")

    tv_values = []

    for i, j in itertools.combinations(range(d), 2):
        size_i, size_j = cardinalities[i], cardinalities[j]

        prob_r = np.zeros((size_i, size_j), dtype=float)
        prob_s = np.zeros((size_i, size_j), dtype=float)

        for row in Xr:
            if 0 <= row[i] < size_i and 0 <= row[j] < size_j:
                prob_r[row[i], row[j]] += 1.0

        for row in Xs:
            if 0 <= row[i] < size_i and 0 <= row[j] < size_j:
                prob_s[row[i], row[j]] += 1.0

        prob_r /= n_real
        prob_s /= n_synth

        tv = 0.5 * np.sum(np.abs(prob_r - prob_s))
        tv_values.append(tv)

    avg_tv = np.mean(tv_values)
    print(f"Average TV distance: {avg_tv:.6f}")

    return {"tv_distance": float(avg_tv)}


def tv_distance_analysis(real_df, synth_df):
    """
    Analyze TV distance with detailed statistics.
    """
    Xr = real_df.values.astype(int)
    Xs = synth_df.values.astype(int)

    n_real, d = Xr.shape
    n_synth = Xs.shape[0]

    print(f"Dataset info:")
    print(f"  Real samples: {n_real}, Synth samples: {n_synth}")
    print(f"  Dimensions: {d}")

    # Method 1: Observed range (your current method)
    cardinalities_observed = [max(Xr[:, i].max(), Xs[:, i].max()) + 1 for i in range(d)]

    tv_values_observed = []
    for i, j in itertools.combinations(range(d), 2):
        size_i, size_j = cardinalities_observed[i], cardinalities_observed[j]

        prob_r = np.zeros((size_i, size_j))
        prob_s = np.zeros((size_i, size_j))

        for row in Xr:
            prob_r[row[i], row[j]] += 1.0
        for row in Xs:
            prob_s[row[i], row[j]] += 1.0

        prob_r /= n_real
        prob_s /= n_synth

        tv = 0.5 * np.sum(np.abs(prob_r - prob_s))
        tv_values_observed.append(tv)

    avg_tv_observed = np.mean(tv_values_observed)

    # Method 2: Assuming domain size might be larger
    # Try adding +1 to each cardinality (common off-by-one issue)
    cardinalities_plus1 = [c + 1 for c in cardinalities_observed]

    tv_values_plus1 = []
    for i, j in itertools.combinations(range(d), 2):
        size_i, size_j = cardinalities_plus1[i], cardinalities_plus1[j]

        prob_r = np.zeros((size_i, size_j))
        prob_s = np.zeros((size_i, size_j))

        for row in Xr:
            prob_r[row[i], row[j]] += 1.0
        for row in Xs:
            prob_s[row[i], row[j]] += 1.0

        prob_r /= n_real
        prob_s /= n_synth

        tv = 0.5 * np.sum(np.abs(prob_r - prob_s))
        tv_values_plus1.append(tv)

    avg_tv_plus1 = np.mean(tv_values_plus1)

    print(f"\nResults:")
    print(f"  Cardinalities (observed): {cardinalities_observed}")
    print(f"  Average TV (observed range): {avg_tv_observed:.6f}")
    print(f"  Cardinalities (+1): {cardinalities_plus1}")
    print(f"  Average TV (+1 to each): {avg_tv_plus1:.6f}")
    print(f"  Paper value: 0.028")
    print(f"  Ratio (observed/paper): {avg_tv_observed / 0.028:.2f}x")

    # Check individual marginal statistics
    print(f"\nMarginal TV statistics:")
    print(f"  Min: {min(tv_values_observed):.6f}")
    print(f"  Max: {max(tv_values_observed):.6f}")
    print(f"  Median: {np.median(tv_values_observed):.6f}")
    print(f"  Std: {np.std(tv_values_observed):.6f}")

    return {
        "tv_distance_observed": float(avg_tv_observed),
        "tv_distance_plus1": float(avg_tv_plus1)
    }
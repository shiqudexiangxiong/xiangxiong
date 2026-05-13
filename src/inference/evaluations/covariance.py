import numpy as np


# def covariance_error(real_df, synth_df):
#     """
#     Compute covariance error as defined in the paper.
#
#     From paper:
#     ||Cov(Emb(D)) - Cov(Emb(D_DP))||_F / ||Cov(Emb(D))||_F
#
#     Where Emb(·) is the embedding function that rescales categorical variables.
#     The paper states this is equivalent to computing covariance on normalized data.
#
#     Args:
#         real_df: Original dataset
#         synth_df: Synthetic DP dataset
#
#     Returns:
#         dict: {"covariance_error": relative Frobenius norm error}
#     """
#     # Convert to numpy arrays
#     Xr = real_df.values if hasattr(real_df, 'values') else np.array(real_df)
#     Xs = synth_df.values if hasattr(synth_df, 'values') else np.array(synth_df)
#
#     # Convert to float
#     Xr = Xr.astype(float)
#     Xs = Xs.astype(float)
#
#     n_real, d = Xr.shape
#     n_synth = Xs.shape[0]
#
#     # Embed/normalize each dimension to [0, 1]
#     # Use combined min/max across both datasets for consistent embedding
#     Xr_emb = np.zeros_like(Xr, dtype=float)
#     Xs_emb = np.zeros_like(Xs, dtype=float)
#
#     for col in range(d):
#         min_val = min(Xr[:, col].min(), Xs[:, col].min())
#         max_val = max(Xr[:, col].max(), Xs[:, col].max())
#
#         if max_val > min_val:
#             Xr_emb[:, col] = (Xr[:, col] - min_val) / (max_val - min_val)
#             Xs_emb[:, col] = (Xs[:, col] - min_val) / (max_val - min_val)
#         else:
#             Xr_emb[:, col] = 0.0
#             Xs_emb[:, col] = 0.0
#
#     # Compute centered covariance matrices
#     # Using unbiased estimator (ddof=1, divide by n-1)
#     cov_real = np.cov(Xr_emb, rowvar=False, ddof=1)
#     cov_synth = np.cov(Xs_emb, rowvar=False, ddof=1)
#
#     # Compute Frobenius norm of difference
#     diff_norm = np.linalg.norm(cov_real - cov_synth, ord='fro')
#
#     # Compute Frobenius norm of real covariance (denominator)
#     real_norm = np.linalg.norm(cov_real, ord='fro')
#
#     # Compute relative error
#     if real_norm > 1e-10:
#         relative_error = diff_norm / real_norm
#     else:
#         relative_error = 0.0
#
#     return {"covariance_error": float(relative_error)}

import numpy as np
def covariance_error(real_df, synth_df, verbose=False, handle_discrete=True):
    """
    Compute covariance error with special handling for discrete data.

    Args:
        real_df: Original dataset
        synth_df: Synthetic DP dataset
        verbose: Print diagnostic information
        handle_discrete: Use special normalization for discrete data

    Returns:
        dict: {"covariance_error": relative Frobenius norm error}
    """
    # Convert to numpy arrays
    Xr = real_df.values if hasattr(real_df, 'values') else np.array(real_df)
    Xs = synth_df.values if hasattr(synth_df, 'values') else np.array(synth_df)

    # Convert to float
    Xr = Xr.astype(float)
    Xs = Xs.astype(float)

    n_real, d = Xr.shape
    n_synth = Xs.shape[0]

    if n_real < 2 or n_synth < 2:
        return {"covariance_error": float('nan')}

    # ✅ 检测是否为离散数据
    is_discrete = [len(np.unique(Xr[:, col])) <= 20 for col in range(d)]

    if verbose and any(is_discrete):
        print(f"Discrete columns: {sum(is_discrete)}/{d}")

    # Embed/normalize each dimension
    Xr_emb = np.zeros_like(Xr, dtype=float)
    Xs_emb = np.zeros_like(Xs, dtype=float)

    for col in range(d):
        min_val = min(Xr[:, col].min(), Xs[:, col].min())
        max_val = max(Xr[:, col].max(), Xs[:, col].max())

        if max_val > min_val:
            if handle_discrete and is_discrete[col]:
                # ✅ 离散列：使用整数规范化
                # 确保离散值映射到均匀网格
                unique_vals = np.unique(np.concatenate([Xr[:, col], Xs[:, col]]))
                n_unique = len(unique_vals)

                # 创建值到索引的映射
                val_to_idx = {v: i for i, v in enumerate(sorted(unique_vals))}

                # 映射到 [0, 1] 的均匀网格
                Xr_emb[:, col] = np.array([val_to_idx[v] for v in Xr[:, col]]) / (n_unique - 1) if n_unique > 1 else 0
                Xs_emb[:, col] = np.array([val_to_idx[v] for v in Xs[:, col]]) / (n_unique - 1) if n_unique > 1 else 0
            else:
                # 连续列：标准规范化
                Xr_emb[:, col] = (Xr[:, col] - min_val) / (max_val - min_val)
                Xs_emb[:, col] = (Xs[:, col] - min_val) / (max_val - min_val)
        else:
            Xr_emb[:, col] = 0.0
            Xs_emb[:, col] = 0.0

    # ✅ 中心化数据（移除均值）
    Xr_centered = Xr_emb - Xr_emb.mean(axis=0)
    Xs_centered = Xs_emb - Xs_emb.mean(axis=0)

    # 计算协方差矩阵（手动计算以更好控制）
    cov_real = (Xr_centered.T @ Xr_centered) / (n_real - 1)
    cov_synth = (Xs_centered.T @ Xs_centered) / (n_synth - 1)

    # ✅ 处理数值误差
    cov_real = np.nan_to_num(cov_real, nan=0.0, posinf=0.0, neginf=0.0)
    cov_synth = np.nan_to_num(cov_synth, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Frobenius norms
    diff_norm = np.linalg.norm(cov_real - cov_synth, ord='fro')
    real_norm = np.linalg.norm(cov_real, ord='fro')

    if verbose:
        print(f"\nCovariance diagnostics:")
        print(f"  Real norm: {real_norm:.6f}")
        print(f"  Diff norm: {diff_norm:.6f}")
        print(f"  Real cov diagonal: {np.diag(cov_real)}")
        print(f"  Synth cov diagonal: {np.diag(cov_synth)}")

    # Compute relative error
    if real_norm > 1e-10:
        relative_error = diff_norm / real_norm
    else:
        relative_error = 0.0

    return {"covariance_error": float(relative_error)}

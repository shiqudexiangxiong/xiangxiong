import pandas as pd
import pickle
import numpy as np
import random
import os
import time
from scipy.stats import truncnorm


def truncated_normal_sample(low, high):
    mu = (low + high) / 2
    sigma = (high - low) / 6

    if sigma == 0:
        return float(low)

    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)


def restore_from_discretized_normal(
    disc_csv_path,
    inverse_mapping_path,
    output_csv_path,
    random_seed=int(time.time())
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    df_disc = pd.read_csv(disc_csv_path)

    with open(inverse_mapping_path, "rb") as f:
        inverse_mapping = pickle.load(f)

    df_restored = df_disc.copy()

    for col in df_disc.columns:
        mapping = inverse_mapping[col]

        def restore_value(x):
            if pd.isna(x):
                return np.nan

            x = int(x)
            val = mapping[x]

            # 连续变量 → 截断正态采样
            if isinstance(val, (tuple, list)) and len(val) == 2:
                low, high = val
                return truncated_normal_sample(low, high)

            # 类别变量 → 精确映射
            else:
                return val

        df_restored[col] = df_disc[col].apply(restore_value)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_restored.to_csv(output_csv_path, index=False)
    print("Restored (normal) data saved to:", output_csv_path)

    return df_restored




if __name__ == "__main__":
    savedir ="../data/datasets/bank/"
    result_data=os.path.join(savedir, "privpgd_synth_uniform_gaussian.csv")
    domain=os.path.join(savedir, "domain.json")
    mapping=os.path.join(savedir, "inverse_mapping.pkl")
    restore_from_discretized_normal(
        disc_csv_path=result_data,
        inverse_mapping_path=mapping,
        output_csv_path=os.path.join(savedir, "privpgd_synth_uniform_gaussian_oringin.csv"),
    )

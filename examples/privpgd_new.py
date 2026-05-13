import csv
import logging
import os
import time
import numpy as np
from .utils_examples import flatten_dict

from src.inference.dataset import Dataset
from src.inference.evaluation import Evaluator
from src.inference.privpgd.inference import AdvancedSlicedInference
from src.mechanisms.KwayWeight import KWayEnhanced
from src.mechanisms.utils_mechanisms import generate_all_kway_workload

from src.inference.evaluations.downstream import downstream_logistic
from src.inference.evaluations.covariance import covariance_error
from src.inference.evaluations.distances import (
    sw1_distance,
    tv_distance_final
)
from src.inference.evaluations.queries import threshold_query_error, counting_query_error

logging.basicConfig(level=logging.INFO)


def run_privpgd(
        savedir="../data/datasets/bank/",
        train_dataset="../data/datasets/baank/data_disc.csv",
        domain="../data/datasets/bank/domain.json",
        epsilon=2.5,
        delta=0.00001,
        weight_strategy="uniform",
        noise_type="gaussian",
        gp_lengthscale=1.0,
        gp_variance=1.0,
        iters=1000,
        n_particles=100000,
        lr=0.1,
        scheduler_step=50,
        scheduler_gamma=0.75,
        num_projections=10,
        scale_reg=0.0,
        p_mask=80,
        batch_size=5,
):
    """
    Run PrivPGD with enhanced privacy mechanisms and weighted budget allocation.

    Args:
        savedir (str): Directory to save results and synthetic data
        train_dataset (str): Path to training dataset CSV
        domain (str): Path to domain JSON file
        epsilon (float): Privacy budget (epsilon)
        delta (float): Privacy parameter delta
        weight_strategy (str): Budget allocation strategy
            Options: "uniform", "variance", "entropy", "importance", "custom"
        noise_type (str): Type of noise mechanism
            Options: "gaussian", "renyi", "gp"
        gp_lengthscale (float): Lengthscale for GP kernel (only for noise_type="gp")
        gp_variance (float): Variance for GP kernel (only for noise_type="gp")
        iters (int): Number of iterations for PrivPGD
        n_particles (int): Number of particles for PrivPGD
        lr (float): Learning rate
        scheduler_step (int): Scheduler step size
        scheduler_gamma (float): Scheduler gamma
        num_projections (int): Number of projections for SW2
        scale_reg (float): Regularization parameter
        p_mask (int): Percentage of masked gradients
        batch_size (int): Batch size

    Returns:
        dict: Dictionary containing experiment results and metrics
    """

    params = locals()

    logging.info("=" * 80)
    logging.info("PRIVPGD WITH ENHANCED PRIVACY MECHANISMS")
    logging.info("=" * 80)
    logging.info(f"Noise mechanism: {noise_type}")
    logging.info(f"Weight strategy: {weight_strategy}")
    logging.info(f"Privacy budget: ε={epsilon}, δ={delta}")

    if noise_type == "gp":
        logging.info(f"GP parameters: lengthscale={gp_lengthscale}, variance={gp_variance}")

    # Load dataset
    logging.info("\nLoading dataset...")
    data = Dataset.load(train_dataset, domain)
    logging.info(f"Dataset: {data.records} records, {len(data.domain)} attributes")

    # Generate workload
    logging.info("\nGenerating 2-way workload...")
    all_2_way_workload = generate_all_kway_workload(data=data, degree=2)
    logging.info(f"Workload size: {len(all_2_way_workload)} marginals")

    # Initialize inference engine
    logging.info("\nInitializing PrivPGD inference engine...")
    generation_engine = AdvancedSlicedInference(
        domain=data.domain,
        hp=params,
    )

    # Initialize mechanism
    logging.info(f"\nInitializing enhanced K-Way mechanism...")

    # Create RDP orders for Rényi DP
    rdp_orders = None
    if noise_type == "renyi":
        rdp_orders = np.concatenate([
            np.linspace(2, 10, 9),
            np.linspace(12, 64, 27)
        ])
        logging.info(f"Using {len(rdp_orders)} RDP orders for privacy accounting")

    mechanism = KWayEnhanced(
        epsilon=epsilon,
        delta=delta,
        degree=2,
        bounded=True,
        weight_strategy=weight_strategy,
        noise_type=noise_type,
        rdp_orders=rdp_orders,
        gp_lengthscale=gp_lengthscale,
        gp_variance=gp_variance,
    )

    # Run mechanism
    logging.info("\nRunning K-Way mechanism with data synthesis...")
    start_time = time.time()
    synth, loss = mechanism.run(
        data=data,
        workload=all_2_way_workload,
        engine=generation_engine,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info(f"\n{'=' * 80}")
    logging.info(f"Synthesis complete!")
    logging.info(f"Total loss: {loss:.4f}")
    logging.info(f"Elapsed time: {elapsed_time:.2f}s")
    logging.info(f"{'=' * 80}\n")

    # Save synthetic data
    if savedir:
        output_file = os.path.join(
            savedir,
            f"privpgd_synth_{weight_strategy}_{noise_type}.csv"
        )
        synth.df.to_csv(output_file, index=False)
        logging.info(f"Synthetic data saved to {output_file}")

    # Evaluation
    logging.info("\nStarting comprehensive evaluation...")
    evaluator = Evaluator(data=data, synth=synth, workload=all_2_way_workload)
    evaluator.set_compression()
    evaluator.update_synth(synth)
    results, _ = evaluator.evaluate()

    # Compute paper metrics
    paper_metrics = {}

    logging.info("  - Computing downstream task metrics...")
    paper_metrics.update(
        downstream_logistic(data.df, synth.df, label_col=data.df.columns[-1])
    )


    logging.info("  - Computing covariance error...")
    paper_metrics.update(covariance_error(data.df, synth.df))

    logging.info("  - Computing query errors...")
    paper_metrics.update(counting_query_error(data.df, synth.df))
    paper_metrics.update(threshold_query_error(data.df, synth.df))

    logging.info("  - Computing SW1 distance...")
    paper_metrics.update(sw1_distance(data.df, synth.df))

    logging.info("  - Computing TV distance...")
    paper_metrics.update(
        tv_distance_final(data.df, synth.df, domain_path=domain)
    )

    # logging.info("  - Analyzing TV distance...")
    # tv_distance_analysis(data.df, synth.df)



    # Compile results
    dataset_name = os.path.basename(os.path.dirname(train_dataset))
    experiment_results = {
        "dataset_name": dataset_name,
        "weight_strategy": weight_strategy,
        "noise_type": noise_type,
        "time": elapsed_time,
        "loss": loss,
        **flatten_dict(
            {
                k: v
                for k, v in params.items()
                if k not in ["savedir", "train_dataset", "domain"]
            }
        ),
        # **flatten_dict(results),
        **flatten_dict(paper_metrics),
    }

    # Save results to CSV
    results_file = os.path.join(savedir, "privpgd_results_enhanced.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    logging.info(f"\nResults saved to {results_file}")
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logging.info("=" * 80)

    return experiment_results


def run_experiments_batch(configs):
    """
    Run multiple experiments with different configurations.

    Args:
        configs (list): List of configuration dictionaries

    Returns:
        list: List of results from each experiment

    Example:
        configs = [
            {"epsilon": 1.0, "weight_strategy": "uniform", "noise_type": "gaussian"},
            {"epsilon": 2.5, "weight_strategy": "importance", "noise_type": "renyi"},
        ]
        results = run_experiments_batch(configs)
    """
    all_results = []

    logging.info("=" * 80)
    logging.info(f"RUNNING BATCH EXPERIMENTS: {len(configs)} configurations")
    logging.info("=" * 80)

    for idx, config in enumerate(configs, 1):
        logging.info(f"\n{'=' * 80}")
        logging.info(f"EXPERIMENT {idx}/{len(configs)}")
        logging.info(f"Configuration: {config}")
        logging.info(f"{'=' * 80}\n")

        try:
            result = run_privpgd(**config)
            all_results.append(result)
        except Exception as e:
            logging.error(f"Experiment {idx} failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("BATCH EXPERIMENTS SUMMARY")
    logging.info("=" * 80)
    successful = sum(1 for r in all_results if r is not None)
    logging.info(f"Successful: {successful}/{len(configs)}")
    logging.info(f"Failed: {len(configs) - successful}/{len(configs)}")

    return all_results


def compare_strategies(
        epsilon_values=[1.0, 2.5],
        weight_strategies=["uniform", "importance"],
        noise_types=["gaussian", "renyi"],
        savedir="../data/datasets/acs_income_CA_2018_default_32/",
        **common_params
):
    """
    Compare different strategies and noise types.

    Args:
        epsilon_values (list): List of epsilon values to test
        weight_strategies (list): List of weight strategies to test
        noise_types (list): List of noise mechanisms to test
        savedir (str): Directory to save results
        **common_params: Common parameters for all experiments

    Returns:
        list: Results from all experiments

    Example:
        results = compare_strategies(
            epsilon_values=[1.0, 2.5],
            weight_strategies=["uniform", "variance", "importance"],
            noise_types=["gaussian", "renyi", "gp"],
            iters=500,  # Common parameter
            n_particles=50000  # Common parameter
        )
    """
    from itertools import product

    # Generate all configurations
    configs = []
    for eps, strategy, noise in product(epsilon_values, weight_strategies, noise_types):
        config = {
            "epsilon": eps,
            "weight_strategy": strategy,
            "noise_type": noise,
            "savedir": savedir,
            **common_params
        }
        configs.append(config)

    logging.info(f"Generated {len(configs)} configurations to test")

    # Run all experiments
    return run_experiments_batch(configs)


def run_single_experiment_quick_test():
    """
    Quick test with reduced parameters for development/testing.

    Returns:
        dict: Experiment results
    """
    logging.info("Running quick test experiment...")

    return run_privpgd(
        epsilon=1.0,
        weight_strategy="importance",
        noise_type="renyi",
        iters=100,  # Reduced for quick test
        n_particles=10000,  # Reduced for quick test
    )


def run_full_comparison():
    """
    Run a comprehensive comparison of all mechanisms.

    This is equivalent to the old compare_mechanisms.py script.

    Returns:
        list: Results from all experiments
    """
    return compare_strategies(
        epsilon_values=[1.0, 2.5, 5.0],
        weight_strategies=["uniform", "variance", "entropy", "importance"],
        noise_types=["gaussian", "renyi", "gp"],
        iters=500,  # Reduced for faster comparison
        n_particles=50000,  # Reduced for faster comparison
    )


def run_baseline_vs_enhanced():
    """
    Compare baseline (Gaussian + Uniform) vs enhanced (Rényi + Importance).

    Returns:
        list: Results from comparison
    """
    configs = [
        {
            "epsilon": 2.5,
            "weight_strategy": "uniform",
            "noise_type": "gaussian",
            "iters": 1000,
            "n_particles": 100000,
        },
        {
            "epsilon": 2.5,
            "weight_strategy": "importance",
            "noise_type": "renyi",
            "iters": 1000,
            "n_particles": 100000,
        },
    ]

    logging.info("Running Baseline vs Enhanced comparison...")
    return run_experiments_batch(configs)


def run_custom_configuration(**kwargs):
    """
    Run experiment with custom configuration.

    Args:
        **kwargs: Any parameters accepted by run_privpgd()

    Returns:
        dict: Experiment results

    Example:
        result = run_custom_configuration(
            epsilon=3.0,
            weight_strategy="variance",
            noise_type="gp",
            gp_lengthscale=2.0,
            iters=800,
        )
    """
    return run_privpgd(**kwargs)


# ============================================================================
# Main execution examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Run a single experiment with default parameters
    print("Example 1: Single experiment with defaults")
    result1 = run_privpgd()

    # Example 2: Run a single experiment with custom parameters
    print("\nExample 2: Single experiment with custom parameters")
    result2 = run_privpgd(
        epsilon=2.5,
        weight_strategy="importance",
        noise_type="renyi",
        iters=1000,
    )

    # Example 3: Quick test
    print("\nExample 3: Quick test")
    # result3 = run_single_experiment_quick_test()

    # Example 4: Baseline vs Enhanced comparison
    print("\nExample 4: Baseline vs Enhanced")
    # results4 = run_baseline_vs_enhanced()

    # Example 5: Compare specific strategies
    print("\nExample 5: Compare strategies")
    # results5 = compare_strategies(
    #     epsilon_values=[2.5],
    #     weight_strategies=["uniform", "importance"],
    #     noise_types=["gaussian", "renyi"],
    #     iters=500,
    #     n_particles=50000,
    # )

    # Example 6: Full comprehensive comparison
    print("\nExample 6: Full comparison")
    # results6 = run_full_comparison()

    # Example 7: Custom batch of experiments
    print("\nExample 7: Custom batch")
    # custom_configs = [
    #     {"epsilon": 1.0, "weight_strategy": "variance", "noise_type": "gaussian"},
    #     {"epsilon": 2.5, "weight_strategy": "entropy", "noise_type": "renyi"},
    #     {"epsilon": 5.0, "weight_strategy": "importance", "noise_type": "gp"},
    # ]
    # results7 = run_experiments_batch(custom_configs)

import csv
import logging
import os
import time
import click
from utils_examples import flatten_dict

from inference.dataset import Dataset
from inference.evaluation import Evaluator
from inference.privpgd.inference import AdvancedSlicedInference
from mechanisms.KwayWeight import KWayWeighted  # Updated import
from mechanisms.utils_mechanisms import generate_all_kway_workload

from inference.evaluations.downstream import downstream_logistic
from inference.evaluations.covariance import covariance_error
from inference.evaluations.distances import (
    sw1_distance,
    tv_distance_final,
    tv_distance_analysis,
    sw1_distance_simple,
)
from inference.evaluations.queries import threshold_query_error, counting_query_error

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--savedir",
    default="../data/datasets/acs_income_CA_2018_default_32/",
    help="Directory to save the generated synthetic dataset.",
)
@click.option(
    "--train_dataset",
    default="../data/datasets/acs_income_CA_2018_default_32/data_disc.csv",
    help="File path for the training dataset (CSV format).",
)
@click.option(
    "--domain",
    default="../data/datasets/acs_income_CA_2018_default_32/domain.json",
    help="File path for the domain description (JSON format).",
)
@click.option(
    "--epsilon",
    default=2.5,
    type=float,
    help="Privacy budget (epsilon) for differential privacy.",
)
@click.option(
    "--delta",
    default=0.00001,
    type=float,
    help="Delta parameter for differential privacy.",
)
@click.option(
    "--weight_strategy",
    default="uniform",
    type=click.Choice(["uniform", "variance", "entropy", "importance", "custom"]),
    help="Strategy for privacy budget allocation: uniform, variance, entropy, importance, or custom.",
)
@click.option(
    "--iters",
    default=1000,
    type=int,
    help="Number of iterations for PrivPGD particle gradient descent.",
)
@click.option(
    "--n_particles",
    default=100000,
    type=int,
    help="Number of particles for PrivPGD.",
)
@click.option("--lr", default=0.1, type=float, help="Learning rate.")
@click.option(
    "--scheduler_step",
    default=50,
    type=float,
    help="Scheduler step size for PrivPGD.",
)
@click.option(
    "--scheduler_gamma",
    default=0.75,
    type=float,
    help="Scheduler gamma (i.e., multiplicative factor) for PrivPGD.",
)
@click.option(
    "--num_projections",
    default=10,
    type=int,
    help="Number of projections to compute SW2 for PrivPGD.",
)
@click.option(
    "--scale_reg",
    default=0.0,
    type=float,
    help="Regularization parameter for constraints in PrivPGD.",
)
@click.option(
    "--p_mask",
    default=80,
    type=int,
    help="Percentage of randomly masked gradients in PrivPGD.",
)
@click.option(
    "--batch_size",
    default=5,
    type=int,
    help="Batch size in PrivPGD.",
)
def run_privpgd(
    savedir,
    train_dataset,
    domain,
    epsilon,
    delta,
    weight_strategy,
    iters,
    n_particles,
    lr,
    scheduler_step,
    scheduler_gamma,
    num_projections,
    scale_reg,
    p_mask,
    batch_size,
):
    """
    Run PrivPGD with weighted privacy budget allocation for privacy-preserving data synthesis.
    """

    params = locals()

    logging.info("Loading dataset...")
    data = Dataset.load(
        params["train_dataset"],
        params["domain"],
    )

    logging.info("Generating 2-way workload...")
    all_2_way_workload = generate_all_kway_workload(data=data, degree=2)
    logging.info(f"Workload size: {len(all_2_way_workload)} marginals")

    logging.info("Initializing PrivPGD inference engine...")
    generation_engine = AdvancedSlicedInference(
        domain=data.domain,
        hp=params,
    )

    logging.info(
        f"Initializing K-Way mechanism with {weight_strategy} weight strategy..."
    )
    mechanism = KWayWeighted(
        epsilon=params["epsilon"],
        delta=params["delta"],
        degree=2,
        bounded=True,
        weight_strategy=weight_strategy,
    )

    logging.info("Running K-Way mechanism...")
    start_time = time.time()
    synth, loss = mechanism.run(
        data=data,
        workload=all_2_way_workload,
        engine=generation_engine,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info(f"Total loss: {loss:.4f}, Elapsed time: {elapsed_time:.2f}s")

    if params["savedir"]:
        output_file = os.path.join(
            params["savedir"],
            f"privpgd_synth_data_{weight_strategy}.csv"
        )
        synth.df.to_csv(output_file, index=False)
        logging.info(f"Synthetic data saved to {output_file}")

    logging.info("Starting evaluation...")
    evaluator = Evaluator(data=data, synth=synth, workload=all_2_way_workload)
    evaluator.set_compression()
    evaluator.update_synth(synth)
    results, _ = evaluator.evaluate()

    paper_metrics = {}

    logging.info("Computing downstream task metrics...")
    paper_metrics.update(
        downstream_logistic(data.df, synth.df, label_col=data.df.columns[-1])
    )

    logging.info("Computing covariance error...")
    paper_metrics.update(covariance_error(data.df, synth.df))

    logging.info("Computing SW1 distance...")
    paper_metrics.update(sw1_distance(data.df, synth.df))

    logging.info("Computing TV distance...")
    paper_metrics.update(
        tv_distance_final(data.df, synth.df, domain_path=params["domain"])
    )

    logging.info("Analyzing TV distance...")
    tv_distance_analysis(data.df, synth.df)

    logging.info("Computing query errors...")
    paper_metrics.update(counting_query_error(data.df, synth.df))
    paper_metrics.update(threshold_query_error(data.df, synth.df))

    dataset_name = os.path.basename(os.path.dirname(train_dataset))
    experiment_results = {
        "dataset_name": dataset_name,
        "weight_strategy": weight_strategy,
        "time": elapsed_time,
        "loss": loss,
        **flatten_dict(
            {
                k: v
                for k, v in params.items()
                if k not in ["savedir", "train_dataset", "domain"]
            }
        ),
        **flatten_dict(results),
        **flatten_dict(paper_metrics),
    }

    # File to save results
    results_file = os.path.join(savedir, "privpgd_results_weighted.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    logging.info(f"Results saved to {results_file}")
    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    run_privpgd()